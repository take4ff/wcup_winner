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
from train_model import expand_df_to_rows, FEATURE_COLS, decay_weights  # noqa: E402
from team_glm import TeamGLM  # noqa: E402

XGB_REG_PARAMS = {
    'objective': 'count:poisson', 'n_estimators': 300, 'learning_rate': 0.05,
    'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'verbosity': 0,
}
XGB_CLS_PARAMS = {
    'objective': 'multi:softprob', 'num_class': 3, 'n_estimators': 300,
    'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'verbosity': 0,
}

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


def walkforward_predictions(df, feature_cols, start_year, end_year, train_start='2015-01-01',
                            half_life=0.0):
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
        w = np.repeat(decay_weights(train_df['date'], half_life,
                                    reference_date=f'{year}-01-01'), 2)

        poisson_pipe = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=500))
        poisson_pipe.fit(X_tr, tr['score'], poissonregressor__sample_weight=w)
        reg = LGBMRegressor(**LGBM_REG_PARAMS)
        reg.fit(X_tr, tr['score'], sample_weight=w)
        cls = LGBMClassifier(**LGBM_CLS_PARAMS)
        cls.fit(X_tr, tr['result_class'], sample_weight=w)

        # 追加成分: チーム攻守GLM（全履歴・時間減衰）
        glm = TeamGLM().fit(df, train_end=f'{year}-01-01')

        # 追加成分: XGBoost回帰・分類器
        from xgboost import XGBRegressor, XGBClassifier
        xreg = XGBRegressor(**XGB_REG_PARAMS)
        xreg.fit(X_tr, tr['score'], sample_weight=w)
        xcls = XGBClassifier(**XGB_CLS_PARAMS)
        xcls.fit(X_tr, tr['result_class'], sample_weight=w)

        # 追加成分: チームIDカテゴリカル入りLGBM（簡易埋め込みの代替）
        team_cats = pd.Categorical(pd.concat([df['home_team'], df['away_team']])).categories
        def team_cols(src_df):
            own = np.empty(2 * len(src_df), dtype=object)
            opp = np.empty(2 * len(src_df), dtype=object)
            own[0::2] = src_df['home_team'].values; opp[0::2] = src_df['away_team'].values
            own[1::2] = src_df['away_team'].values; opp[1::2] = src_df['home_team'].values
            return (pd.Categorical(own, categories=team_cats).codes,
                    pd.Categorical(opp, categories=team_cats).codes)
        own_tr, opp_tr = team_cols(train_df)
        X_tr_cat = X_tr.copy(); X_tr_cat['own_id'] = own_tr; X_tr_cat['opp_id'] = opp_tr
        creg = LGBMRegressor(**LGBM_REG_PARAMS)
        creg.fit(X_tr_cat, tr['score'], sample_weight=w,
                 categorical_feature=['own_id', 'opp_id'])

        ev = expand_df_to_rows(eval_df)
        X_ev = ev[feature_cols]
        lam_p = poisson_pipe.predict(X_ev)
        lam_l = reg.predict(X_ev)
        lam_x = xreg.predict(X_ev)
        own_ev, opp_ev = team_cols(eval_df)
        X_ev_cat = X_ev.copy(); X_ev_cat['own_id'] = own_ev; X_ev_cat['opp_id'] = opp_ev
        lam_c = creg.predict(X_ev_cat)
        probas = cls.predict_proba(X_ev)
        xprobas = xcls.predict_proba(X_ev)

        # GLMのλ（ホーム視点行はwas_home= not neutral）
        was_home = (~eval_df['neutral'].astype(bool)).astype(int).values
        lam_g_h = glm.predict_lambdas(eval_df['home_team'], eval_df['away_team'], was_home)
        lam_g_a = glm.predict_lambdas(eval_df['away_team'], eval_df['home_team'], np.zeros(len(eval_df)))

        # 偶数行=ホーム視点, 奇数行=アウェイ視点
        for i, (_, row) in enumerate(eval_df.iterrows()):
            if row['home_score'] > row['away_score']:
                actual = 'H'
            elif row['home_score'] < row['away_score']:
                actual = 'A'
            else:
                actual = 'D'
            records.append({
                'year': year, 'date': row['date'],
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'home_score': row['home_score'], 'away_score': row['away_score'],
                'lambda_home': (lam_p[2 * i] + lam_l[2 * i]) / 2.0,
                'lambda_away': (lam_p[2 * i + 1] + lam_l[2 * i + 1]) / 2.0,
                'lam_p_h': lam_p[2 * i], 'lam_p_a': lam_p[2 * i + 1],
                'lam_l_h': lam_l[2 * i], 'lam_l_a': lam_l[2 * i + 1],
                'lam_x_h': lam_x[2 * i], 'lam_x_a': lam_x[2 * i + 1],
                'lam_c_h': lam_c[2 * i], 'lam_c_a': lam_c[2 * i + 1],
                'lam_g_h': lam_g_h[i], 'lam_g_a': lam_g_a[i],
                'p_home_cls': probas[2 * i][2], 'p_draw_cls': probas[2 * i][1],
                'p_away_cls': probas[2 * i][0],
                'p_home_xcls': xprobas[2 * i][2], 'p_draw_xcls': xprobas[2 * i][1],
                'p_away_xcls': xprobas[2 * i][0],
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

    print(f"Running walk-forward predictions (half_life={args.half_life})...")
    preds = walkforward_predictions(df, feature_cols, args.start_year, args.end_year,
                                    half_life=args.half_life)
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


def _outcome_probs(lh_arr, la_arr, rho):
    """λ配列 → [pH,pD,pA] 行列（Dixon-Coles補正つき）"""
    out = np.empty((len(lh_arr), 3))
    for i, (lh, la) in enumerate(zip(lh_arr, la_arr)):
        pm = dixon_coles_matrix(lh, la, rho)
        out[i] = [np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))]
    return out


def _logloss(p, y):
    eps = 1e-15
    p = np.clip(p / p.sum(axis=1, keepdims=True), eps, 1 - eps)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def mode_ensemble(args):
    """λアンサンブル構成と分類器ブレンド構成の比較（rho/blendは現行最適値で固定）"""
    preds = pd.read_csv(os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv"))
    y = preds['actual'].map({'H': 0, 'D': 1, 'A': 2}).values
    rho, w_cls = -0.09, 0.25

    lam_configs = {
        'P+L (現行)':   (preds[['lam_p_h', 'lam_l_h']].mean(1), preds[['lam_p_a', 'lam_l_a']].mean(1)),
        'P+L+G':       (preds[['lam_p_h', 'lam_l_h', 'lam_g_h']].mean(1), preds[['lam_p_a', 'lam_l_a', 'lam_g_a']].mean(1)),
        'P+L+X':       (preds[['lam_p_h', 'lam_l_h', 'lam_x_h']].mean(1), preds[['lam_p_a', 'lam_l_a', 'lam_x_a']].mean(1)),
        'P+L+C':       (preds[['lam_p_h', 'lam_l_h', 'lam_c_h']].mean(1), preds[['lam_p_a', 'lam_l_a', 'lam_c_a']].mean(1)),
        'P+L+G+X':     (preds[['lam_p_h', 'lam_l_h', 'lam_g_h', 'lam_x_h']].mean(1), preds[['lam_p_a', 'lam_l_a', 'lam_g_a', 'lam_x_a']].mean(1)),
        'P+L+G+X+C':   (preds[['lam_p_h', 'lam_l_h', 'lam_g_h', 'lam_x_h', 'lam_c_h']].mean(1), preds[['lam_p_a', 'lam_l_a', 'lam_g_a', 'lam_x_a', 'lam_c_a']].mean(1)),
        'G単体':        (preds['lam_g_h'], preds['lam_g_a']),
        'C単体':        (preds['lam_c_h'], preds['lam_c_a']),
        'X単体':        (preds['lam_x_h'], preds['lam_x_a']),
    }
    cls_l = preds[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
    cls_lx = (cls_l + preds[['p_home_xcls', 'p_draw_xcls', 'p_away_xcls']].values) / 2.0

    print(f"=== λアンサンブル構成の比較 (rho={rho}, cls_blend={w_cls}, n={len(preds)}) ===")
    print(f"{'構成':<14} {'cls=LGBM':>10} {'cls=LGBM+XGB':>14}")
    results = {}
    for name, (lh, la) in lam_configs.items():
        mat = _outcome_probs(lh.values, la.values, rho)
        ll_l = _logloss((1 - w_cls) * mat + w_cls * cls_l, y)
        ll_lx = _logloss((1 - w_cls) * mat + w_cls * cls_lx, y)
        results[name] = min(ll_l, ll_lx)
        print(f"{name:<14} {ll_l:>10.5f} {ll_lx:>14.5f}")
    best = min(results, key=results.get)
    print(f"\nBest λ構成: {best} (LogLoss={results[best]:.5f})")


def bivariate_poisson_matrix(lh, la, lam3, max_goals=10):
    """二変量Poisson: X=X1+X3, Y=X2+X3, λ1=lh-λ3, λ2=la-λ3"""
    from math import comb, factorial
    l1, l2 = max(lh - lam3, 0.05), max(la - lam3, 0.05)
    base = np.exp(-(l1 + l2 + lam3))
    pm = np.zeros((max_goals + 1, max_goals + 1))
    ratio = lam3 / (l1 * l2) if lam3 > 0 else 0.0
    for x in range(max_goals + 1):
        for ycol in range(max_goals + 1):
            s = sum(comb(x, k) * comb(ycol, k) * factorial(k) * ratio ** k
                    for k in range(0, min(x, ycol) + 1)) if lam3 > 0 else 1.0
            pm[x, ycol] = base * l1 ** x / factorial(x) * l2 ** ycol / factorial(ycol) * s
    return pm / pm.sum()


def mode_matrix(args):
    """スコア行列モデルの比較: Dixon-Coles(ρ) vs 二変量Poisson(λ3)。
    1X2 Log Loss と 正確なスコアの Log Loss の両方で評価"""
    preds = pd.read_csv(os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv"))
    y = preds['actual'].map({'H': 0, 'D': 1, 'A': 2}).values
    hs = preds['home_score'].clip(upper=MAX_GOALS).astype(int).values
    as_ = preds['away_score'].clip(upper=MAX_GOALS).astype(int).values
    lh, la = preds['lambda_home'].values, preds['lambda_away'].values
    cls = preds[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
    w_cls, eps = 0.25, 1e-15

    print(f"=== スコア行列の比較 (λ=P+L, n={len(preds)}) ===")
    print(f"{'モデル':<16} {'1X2 LL(blend後)':>16} {'スコアLL':>10}")
    for label, builder, grid in [
        ('DC rho', lambda a, b, v: dixon_coles_matrix(a, b, v), [-0.12, -0.09, -0.06, 0.0]),
        ('BP lam3', bivariate_poisson_matrix, [0.05, 0.1, 0.15, 0.2]),
    ]:
        for v in grid:
            mat3 = np.empty((len(preds), 3))
            score_ll = 0.0
            for i in range(len(preds)):
                pm = builder(lh[i], la[i], v)
                mat3[i] = [np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))]
                score_ll -= np.log(max(pm[hs[i], as_[i]], eps))
            ll_1x2 = _logloss((1 - w_cls) * mat3 + w_cls * cls, y)
            print(f"{label} = {v:<8} {ll_1x2:>16.5f} {score_ll / len(preds):>10.5f}")


def mode_stack(args):
    """スタッキング: 成分確率を入力にした多項ロジスティックで最終1X2確率を合成。
    年次ウォークフォワード（その年より前の予測で学習）で固定ブレンドと比較"""
    from sklearn.linear_model import LogisticRegression
    preds = pd.read_csv(os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv"))
    y = preds['actual'].map({'H': 0, 'D': 1, 'A': 2}).values
    rho = -0.09

    mat = _outcome_probs(preds['lambda_home'].values, preds['lambda_away'].values, rho)
    cls = preds[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
    xcls = preds[['p_home_xcls', 'p_draw_xcls', 'p_away_xcls']].values
    glm_mat = _outcome_probs(preds['lam_g_h'].values, preds['lam_g_a'].values, rho)

    eps = 1e-9
    X_meta = np.column_stack([
        np.log(np.clip(mat, eps, 1)), np.log(np.clip(cls, eps, 1)),
        np.log(np.clip(xcls, eps, 1)), np.log(np.clip(glm_mat, eps, 1)),
        (preds['lambda_home'] - preds['lambda_away']).values,
        (preds['lambda_home'] + preds['lambda_away']).values,
    ])
    years = preds['year'].values

    fixed = (1 - 0.25) * mat + 0.25 * cls
    stack_p = np.full((len(preds), 3), np.nan)
    eval_years = sorted(set(years))[2:]  # 最初の2年は学習専用
    for yr in eval_years:
        tr_idx, ev_idx = years < yr, years == yr
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(X_meta[tr_idx], y[tr_idx])
        stack_p[ev_idx] = meta.predict_proba(X_meta[ev_idx])

    mask = ~np.isnan(stack_p[:, 0])
    print(f"=== スタッキング評価 (評価対象 {mask.sum()} 試合, {eval_years[0]}年以降) ===")
    print(f"  固定ブレンド (現行 0.75行列+0.25分類器): LogLoss {_logloss(fixed[mask], y[mask]):.5f}")
    print(f"  スタッキング (多項ロジスティック)      : LogLoss {_logloss(stack_p[mask], y[mask]):.5f}")
    wc_mask = mask & preds['is_wc'].values
    if wc_mask.sum():
        print(f"  [W杯のみ n={wc_mask.sum()}] 固定 {_logloss(fixed[wc_mask], y[wc_mask]):.5f} "
              f"vs スタック {_logloss(stack_p[wc_mask], y[wc_mask]):.5f}")


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
    parser.add_argument('--mode', choices=['probs', 'shrink', 'elo', 'ensemble', 'matrix', 'stack'],
                        default='probs')
    parser.add_argument('--start_year', type=int, default=2019)
    parser.add_argument('--end_year', type=int, default=2026)
    parser.add_argument('--rho', type=float, default=-0.03, help='shrinkモードで使うρ')
    parser.add_argument('--cls_blend', type=float, default=0.5, help='shrinkモードで使うブレンド比')
    parser.add_argument('--half_life', type=float, default=0.0,
                        help='時間減衰サンプル重みの半減期(年)。0で無効')
    args = parser.parse_args()

    if args.mode == 'probs':
        mode_probs(args)
    elif args.mode == 'shrink':
        mode_shrink(args)
    elif args.mode == 'ensemble':
        mode_ensemble(args)
    elif args.mode == 'matrix':
        mode_matrix(args)
    elif args.mode == 'stack':
        mode_stack(args)
    else:
        mode_elo(args)


if __name__ == "__main__":
    main()
