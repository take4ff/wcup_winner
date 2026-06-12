#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
walkforward.py - 時系列ウォークフォワード検証とパラメータ調整

W杯2大会(64試合×2)だけでは差が小さいパラメータの調整はノイズに埋もれるため、
全国際試合を対象に「その年より前のデータだけで学習 → その年を予測」を繰り返し、
大きなサンプルでパラメータを評価する。

モード:
  --mode probs  : Dixon-Coles ρ × 分類器ブレンド比のグリッド評価 (デフォルト)
  --mode shrink : 市場シュリンク比の評価 (2018/2022 W杯のオッズを使用)
  --mode elo    : Elo の Kスケール × ホーム補正のグリッド評価

使い方:
  python src/backtest/walkforward.py --mode probs --start_year 2019 --end_year 2026
  python src/backtest/walkforward.py --mode elo
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pipeline"))
from train_model import expand_df_to_rows, FEATURE_COLS  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MAX_GOALS = 10

# 本番モデルのOptuna最適値を固定ハイパラとして使用（毎年のOptunaは重すぎるため）
LGBM_REG_PARAMS = {
    'objective': 'poisson', 'n_estimators': 212, 'learning_rate': 0.0719,
    'max_depth': 3, 'num_leaves': 18, 'min_child_samples': 43,
    'subsample': 0.629, 'colsample_bytree': 0.517,
    'random_state': 42, 'verbosity': -1,
}
LGBM_CLS_PARAMS = {
    'objective': 'multiclass', 'num_class': 3, 'n_estimators': 73,
    'learning_rate': 0.0677, 'max_depth': 4, 'num_leaves': 63,
    'min_child_samples': 80, 'subsample': 0.703, 'colsample_bytree': 0.716,
    'random_state': 42, 'verbosity': -1,
}


def get_feature_cols():
    return FEATURE_COLS


def dixon_coles_matrix(lh, la, rho):
    goals = np.arange(MAX_GOALS + 1)
    pm = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    pm /= pm.sum()
    if abs(rho) > 1e-9:
        pm[0, 0] *= max(1.0 - lh * la * rho, 0.0)
        pm[1, 0] *= max(1.0 + la * rho, 0.0)
        pm[0, 1] *= max(1.0 + lh * rho, 0.0)
        pm[1, 1] *= max(1.0 - rho, 0.0)
        pm /= pm.sum()
    return pm


def walkforward_predictions(df, feature_cols, start_year, end_year, train_start='2015-01-01'):
    """年次ウォークフォワードで全評価試合の (λh, λa, 分類器確率, 実結果) を生成"""
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from lightgbm import LGBMRegressor, LGBMClassifier

    records = []
    for year in range(start_year, end_year + 1):
        train_df = df[(df['date'] >= train_start) & (df['date'] < f'{year}-01-01')]
        eval_df = df[(df['date'] >= f'{year}-01-01') & (df['date'] < f'{year + 1}-01-01')]
        if len(eval_df) == 0:
            continue
        print(f"  [{year}] train={len(train_df)} eval={len(eval_df)}")

        tr = expand_df_to_rows(train_df)
        X_tr = tr[feature_cols]

        poisson_pipe = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=500))
        poisson_pipe.fit(X_tr, tr['score'])
        reg = LGBMRegressor(**LGBM_REG_PARAMS)
        reg.fit(X_tr, tr['score'])
        cls = LGBMClassifier(**LGBM_CLS_PARAMS)
        cls.fit(X_tr, tr['result_class'])

        ev = expand_df_to_rows(eval_df)
        X_ev = ev[feature_cols]
        lambdas = (poisson_pipe.predict(X_ev) + reg.predict(X_ev)) / 2.0
        probas = cls.predict_proba(X_ev)  # own視点 [負, 分, 勝]

        # 偶数行=ホーム視点, 奇数行=アウェイ視点
        n = len(eval_df)
        for i, (_, row) in enumerate(eval_df.iterrows()):
            lh, la = lambdas[2 * i], lambdas[2 * i + 1]
            p_cls = probas[2 * i]  # ホーム視点: [away勝, 分, home勝]
            if row['home_score'] > row['away_score']:
                actual = 'H'
            elif row['home_score'] < row['away_score']:
                actual = 'A'
            else:
                actual = 'D'
            records.append({
                'year': year, 'date': row['date'],
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'lambda_home': lh, 'lambda_away': la,
                'p_home_cls': p_cls[2], 'p_draw_cls': p_cls[1], 'p_away_cls': p_cls[0],
                'actual': actual,
                'is_wc': row['tournament'] == 'FIFA World Cup',
            })
    return pd.DataFrame(records)


def evaluate_grid(preds, rho_candidates, blend_candidates):
    """ρ × ブレンド比 のグリッドで Log Loss / Brier を評価"""
    eps = 1e-15
    y = preds['actual'].map({'H': 0, 'D': 1, 'A': 2}).values
    results = []
    for rho in rho_candidates:
        mat_probs = np.array([
            [np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))]
            for pm in (dixon_coles_matrix(lh, la, rho)
                       for lh, la in zip(preds['lambda_home'], preds['lambda_away']))
        ])
        cls_probs = preds[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
        for w in blend_candidates:
            p = (1 - w) * mat_probs + w * cls_probs
            p = p / p.sum(axis=1, keepdims=True)
            p = np.clip(p, eps, 1 - eps)
            ll = -np.mean(np.log(p[np.arange(len(y)), y]))
            onehot = np.eye(3)[y]
            brier = np.mean(np.sum((p - onehot) ** 2, axis=1))
            results.append({'rho': rho, 'cls_blend': w, 'log_loss': ll, 'brier': brier})
    return pd.DataFrame(results)


def mode_probs(args):
    df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/features.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    feature_cols = get_feature_cols()

    print("Running walk-forward predictions...")
    preds = walkforward_predictions(df, feature_cols, args.start_year, args.end_year)
    preds_path = os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv")
    preds.to_csv(preds_path, index=False)
    print(f"Saved {len(preds)} match predictions to {preds_path}")

    rho_candidates = [-0.15, -0.12, -0.09, -0.06, -0.03, 0.0]
    blend_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n=== 全評価試合でのグリッド評価 ===")
    grid = evaluate_grid(preds, rho_candidates, blend_candidates)
    grid_path = os.path.join(BASE_DIR, "data/processed/walkforward_grid.csv")
    grid.to_csv(grid_path, index=False)
    best = grid.sort_values('log_loss').iloc[0]
    print(grid.pivot(index='rho', columns='cls_blend', values='log_loss').round(5).to_string())
    print(f"\nBest: rho={best['rho']:.2f}, cls_blend={best['cls_blend']:.2f} "
          f"(LogLoss={best['log_loss']:.5f}, Brier={best['brier']:.5f})")

    wc = preds[preds['is_wc']]
    if len(wc) > 0:
        print(f"\n=== W杯本大会のみ ({len(wc)}試合) ===")
        grid_wc = evaluate_grid(wc, rho_candidates, blend_candidates)
        best_wc = grid_wc.sort_values('log_loss').iloc[0]
        print(grid_wc.pivot(index='rho', columns='cls_blend', values='log_loss').round(5).to_string())
        print(f"\nBest(W杯): rho={best_wc['rho']:.2f}, cls_blend={best_wc['cls_blend']:.2f} "
              f"(LogLoss={best_wc['log_loss']:.5f})")


def mode_shrink(args):
    """市場シュリンク比の評価: ウォークフォワード予測のうち2018/2022 W杯試合をオッズと突合"""
    preds_path = os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv")
    if not os.path.exists(preds_path):
        print("[ERROR] 先に --mode probs を実行してください。")
        sys.exit(1)
    preds = pd.read_csv(preds_path)
    preds = preds[preds['is_wc']].copy()

    odds_frames = []
    for path in ["data/raw/odds/odds_russia2018.csv", "data/raw/odds/odds_qatar2022.csv"]:
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            odds_frames.append(pd.read_csv(full))
    odds = pd.concat(odds_frames, ignore_index=True)

    def key(h, a):
        return f"{min(h, a)}_{max(h, a)}"

    odds['key'] = odds.apply(lambda r: key(r['home_team'], r['away_team']), axis=1)
    preds['key'] = preds.apply(lambda r: key(r['home_team'], r['away_team']), axis=1)
    merged = preds.merge(
        odds[['key', 'home_team', 'odds_home', 'odds_draw', 'odds_away']]
            .rename(columns={'home_team': 'odds_home_team'}).drop_duplicates('key'),
        on='key', how='inner')
    flipped = merged['odds_home_team'] != merged['home_team']
    merged.loc[flipped, ['odds_home', 'odds_away']] = merged.loc[flipped, ['odds_away', 'odds_home']].values
    print(f"オッズ突合済みW杯試合: {len(merged)} (反転補正 {flipped.sum()})")

    # モデル確率 (probsモードのベスト設定を引数で渡す)
    eps = 1e-15
    y = merged['actual'].map({'H': 0, 'D': 1, 'A': 2}).values
    mat = np.array([
        [np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))]
        for pm in (dixon_coles_matrix(lh, la, args.rho)
                   for lh, la in zip(merged['lambda_home'], merged['lambda_away']))
    ])
    cls_p = merged[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
    model_p = (1 - args.cls_blend) * mat + args.cls_blend * cls_p
    model_p /= model_p.sum(axis=1, keepdims=True)

    imp = 1.0 / merged[['odds_home', 'odds_draw', 'odds_away']].values
    imp /= imp.sum(axis=1, keepdims=True)

    print(f"\n=== 市場シュリンク比 (rho={args.rho}, cls_blend={args.cls_blend}) ===")
    print(f"{'w_market':>9} {'LogLoss':>9} {'Brier':>9}")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        p = np.clip((1 - w) * model_p + w * imp, eps, 1 - eps)
        p /= p.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(p[np.arange(len(y)), y]))
        brier = np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))
        print(f"{w:>9.1f} {ll:>9.5f} {brier:>9.5f}")


def mode_elo(args):
    """Elo の Kスケール × ホーム補正のグリッド評価 (期待スコアのBrier)"""
    sys.path.insert(0, os.path.join(BASE_DIR, "src/pipeline"))
    from elo import get_k_factor, get_goal_difference_multiplier
    from collections import defaultdict

    df = pd.read_csv(os.path.join(BASE_DIR, "data/raw/match/results.csv"))
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 評価は2000年以降の試合のみ（それ以前はバーンイン）
    eval_mask = (df['date'] >= '2000-01-01').values

    rows = list(df[['home_team', 'away_team', 'home_score', 'away_score',
                    'tournament', 'neutral']].itertuples(index=False))
    k_factors = np.array([get_k_factor(r.tournament) for r in rows])
    mults = np.array([get_goal_difference_multiplier(r.home_score, r.away_score) for r in rows])
    actuals = np.array([1.0 if r.home_score > r.away_score else
                        (0.0 if r.home_score < r.away_score else 0.5) for r in rows])
    neutrals = np.array([bool(r.neutral) for r in rows])

    print(f"{'k_scale':>8} {'home_adv':>9} {'Brier':>9} {'LogLoss(引分除く)':>16}")
    results = []
    for k_scale in [0.6, 0.8, 1.0, 1.2, 1.4]:
        for home_adv in [50, 75, 100, 125, 150]:
            elo = defaultdict(lambda: 1500.0)
            briers, lls, n_ll = 0.0, 0.0, 0
            n_eval = 0
            for i, r in enumerate(rows):
                rh, ra = elo[r.home_team], elo[r.away_team]
                h_adv = 0 if neutrals[i] else home_adv
                exp_h = 1.0 / (1.0 + 10.0 ** (-((rh + h_adv) - ra) / 400.0))
                if eval_mask[i]:
                    briers += (exp_h - actuals[i]) ** 2
                    n_eval += 1
                    if actuals[i] != 0.5:
                        p = exp_h if actuals[i] == 1.0 else 1.0 - exp_h
                        lls += -np.log(max(p, 1e-15))
                        n_ll += 1
                delta = k_scale * k_factors[i] * mults[i] * (actuals[i] - exp_h)
                elo[r.home_team] = rh + delta
                elo[r.away_team] = ra - delta
            brier = briers / n_eval
            ll = lls / n_ll
            results.append({'k_scale': k_scale, 'home_adv': home_adv, 'brier': brier, 'log_loss': ll})
            print(f"{k_scale:>8.1f} {home_adv:>9d} {brier:>9.5f} {ll:>16.5f}")

    res = pd.DataFrame(results)
    best = res.sort_values('brier').iloc[0]
    print(f"\nBest: k_scale={best['k_scale']}, home_adv={int(best['home_adv'])} "
          f"(Brier={best['brier']:.5f})")
    res.to_csv(os.path.join(BASE_DIR, "data/processed/elo_grid.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description='ウォークフォワード検証・パラメータ調整')
    parser.add_argument('--mode', choices=['probs', 'shrink', 'elo'], default='probs')
    parser.add_argument('--start_year', type=int, default=2019)
    parser.add_argument('--end_year', type=int, default=2026)
    parser.add_argument('--rho', type=float, default=-0.03, help='shrinkモードで使うρ')
    parser.add_argument('--cls_blend', type=float, default=0.5, help='shrinkモードで使うブレンド比')
    args = parser.parse_args()

    if args.mode == 'probs':
        mode_probs(args)
    elif args.mode == 'shrink':
        mode_shrink(args)
    else:
        mode_elo(args)


if __name__ == "__main__":
    main()
