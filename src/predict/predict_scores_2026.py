"""
predict_scores_2026.py
2026年FIFAワールドカップ グループステージ全48試合 + ノックアウト想定試合の
最確スコア・期待得点・勝敗確率を計算してCSV出力・コンソール表示する。
"""
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator_2026 import (
    GROUPS_2026, load_team_features, make_match_features, precompute_all_lambdas,
    group_fixture_rows
)


RHO = -0.09      # Dixon-Coles 補正パラメータ (walkforward 2019-2026 全試合で最適化)
MAX_GOALS = 10
CLS_BLEND = 0.25  # 1X2確率における LGBMClassifier のブレンド比率 (walkforwardで最適化)


def score_prob_matrix(lambda_a, lambda_b, rho=RHO):
    """Dixon-Coles補正済み同時確率行列を返す"""
    goals = np.arange(MAX_GOALS + 1)
    pm = np.outer(poisson.pmf(goals, lambda_a), poisson.pmf(goals, lambda_b))
    pm /= pm.sum()
    if abs(rho) > 1e-9:
        pm[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 0.0)
        pm[1, 0] *= max(1.0 + lambda_b * rho, 0.0)
        pm[0, 1] *= max(1.0 + lambda_a * rho, 0.0)
        pm[1, 1] *= max(1.0 - rho, 0.0)
        pm /= pm.sum()
    return pm


def precompute_all_cls_probas(team_feats, model_classifier, feature_cols, model_xgb_cls=None):
    """全対戦カードの分類器確率 (own視点: 0=負, 1=分, 2=勝) を一括計算してキャッシュ。
    XGBoost分類器が渡された場合はLGBMとの平均を使う"""
    def predict(X):
        proba = model_classifier.predict_proba(X)
        if model_xgb_cls is not None:
            proba = (proba + model_xgb_cls.predict_proba(X)) / 2.0
        return proba

    all_teams = list(team_feats.keys())
    pairs, rows = [], []
    for t1 in all_teams:
        for t2 in all_teams:
            if t1 != t2:
                pairs.append((t1, t2))
                rows.append(make_match_features(t1, t2, team_feats))

    X = pd.DataFrame(rows)[feature_cols]
    cache = {pair: proba for pair, proba in zip(pairs, predict(X))}

    # グループステージの実日程分は、実開催地の標高・休養日数で再計算して上書き
    fx_pairs, fx_rows = group_fixture_rows(team_feats)
    if fx_pairs:
        X_fx = pd.DataFrame(fx_rows)[feature_cols]
        for p, proba in zip(fx_pairs, predict(X_fx)):
            cache[p] = proba
    return cache


def predict_match_score(team_a, team_b, lambda_cache, cls_proba_cache=None,
                        rho=RHO, cls_blend=CLS_BLEND):
    """最確スコア・期待得点・勝敗確率を返す"""
    la = lambda_cache[(team_a, team_b)]
    lb = lambda_cache[(team_b, team_a)]
    pm = score_prob_matrix(la, lb, rho)

    idx = np.argmax(pm)
    pred_a = idx // (MAX_GOALS + 1)
    pred_b = idx % (MAX_GOALS + 1)
    pred_prob = pm[pred_a, pred_b]

    p_win_a = float(np.sum(np.tril(pm, -1)))
    p_draw  = float(np.sum(np.diag(pm)))
    p_win_b = float(np.sum(np.triu(pm, 1)))

    # 分類器確率とのブレンド (backtestと同じ構成)
    if cls_proba_cache is not None and cls_blend > 0:
        proba = cls_proba_cache[(team_a, team_b)]  # own=team_a視点 [負, 分, 勝]
        p_win_a = (1 - cls_blend) * p_win_a + cls_blend * float(proba[2])
        p_draw  = (1 - cls_blend) * p_draw  + cls_blend * float(proba[1])
        p_win_b = (1 - cls_blend) * p_win_b + cls_blend * float(proba[0])
        total = p_win_a + p_draw + p_win_b
        p_win_a, p_draw, p_win_b = p_win_a / total, p_draw / total, p_win_b / total

    return {
        'pred_a': int(pred_a), 'pred_b': int(pred_b),
        'pred_score_prob': round(pred_prob * 100, 1),
        'lambda_a': round(la, 3), 'lambda_b': round(lb, 3),
        'p_win_a': round(p_win_a * 100, 1),
        'p_draw':  round(p_draw  * 100, 1),
        'p_win_b': round(p_win_b * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='2026年W杯 グループステージ スコア予測')
    parser.add_argument('--cls_blend', type=float, default=CLS_BLEND,
                        help='1X2確率における LGBMClassifier のブレンド比率 (default: 0.5)')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path     = os.path.join(base_dir, "data/processed/features.csv")
    poisson_model_path = os.path.join(base_dir, "models/poisson_model.joblib")
    lgbm_model_path   = os.path.join(base_dir, "models/lgbm_model.joblib")
    lgbm_classifier_path = os.path.join(base_dir, "models/lgbm_classifier_model.joblib")
    feature_cols_path  = os.path.join(base_dir, "models/feature_cols.joblib")
    output_path        = os.path.join(base_dir, "data/processed/2026/predicted_scores.csv")

    print("Loading models and features...")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    model_poisson = joblib.load(poisson_model_path)
    model_lgbm    = joblib.load(lgbm_model_path)
    model_classifier = joblib.load(lgbm_classifier_path)
    feature_cols  = joblib.load(feature_cols_path)
    models_dir = os.path.dirname(poisson_model_path)
    model_lgbm_cat = joblib.load(os.path.join(models_dir, "lgbm_cat_model.joblib"))
    model_xgb_cls = joblib.load(os.path.join(models_dir, "xgb_classifier_model.joblib"))
    team_categories = joblib.load(os.path.join(models_dir, "team_categories.joblib"))

    print("Loading team features...")
    team_feats = load_team_features(df)

    print("Precomputing lambdas...")
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols,
                                          model_lgbm_cat, team_categories)

    print(f"Precomputing classifier probabilities (blend={args.cls_blend})...")
    cls_proba_cache = precompute_all_cls_probas(team_feats, model_classifier, feature_cols,
                                                model_xgb_cls)

    # グループステージ全試合の予測
    records = []
    print("\n" + "=" * 90)
    print(f"{'GRP':<4} {'Home':<20} {'Pred':^7} {'Away':<20} "
          f"{'λH':>5} {'λA':>5}  {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'MostLikelyProb':>14}")
    print("=" * 90)

    for grp in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[grp]
        print(f"\n--- Group {grp} ---")
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                res = predict_match_score(home, away, lambda_cache,
                                          cls_proba_cache=cls_proba_cache,
                                          cls_blend=args.cls_blend)
                score_str = f"{res['pred_a']}-{res['pred_b']}"
                print(f" {grp:<4} {home:<20} {score_str:^7} {away:<20} "
                      f"{res['lambda_a']:>5.2f} {res['lambda_b']:>5.2f}  "
                      f"{res['p_win_a']:>5.1f}% {res['p_draw']:>5.1f}% {res['p_win_b']:>5.1f}%  "
                      f"{res['pred_score_prob']:>5.1f}%")
                records.append({
                    'group': grp, 'home_team': home, 'away_team': away,
                    'pred_home_score': res['pred_a'],
                    'pred_away_score': res['pred_b'],
                    'pred_score_str':  f"{res['pred_a']}-{res['pred_b']}",
                    'pred_score_prob': res['pred_score_prob'],
                    'lambda_home': res['lambda_a'],
                    'lambda_away': res['lambda_b'],
                    'p_home_win': res['p_win_a'],
                    'p_draw':     res['p_draw'],
                    'p_away_win': res['p_win_b'],
                })

    print("\n" + "=" * 90)

    # === R32 決勝トーナメント（確定対戦カード）の予測 ===
    R32_FIXTURES = [
        # (home, away, date, FIFA_match_no)
        ('South Africa',            'Canada',                   '2026-06-28', 73),
        ('Brazil',                  'Japan',                    '2026-06-29', 76),
        ('Germany',                 'Paraguay',                 '2026-06-29', 74),
        ('Netherlands',             'Morocco',                  '2026-06-29', 75),
        ('France',                  'Sweden',                   '2026-06-30', 77),
        ('Ivory Coast',             'Norway',                   '2026-06-30', 78),
        ('Mexico',                  'Ecuador',                  '2026-06-30', 79),
        ('England',                 'DR Congo',                 '2026-07-01', 80),
        ('United States',           'Bosnia and Herzegovina',   '2026-07-01', 81),
        ('Belgium',                 'Senegal',                  '2026-07-01', 82),
        ('Portugal',                'Croatia',                  '2026-07-02', 83),
        ('Spain',                   'Austria',                  '2026-07-02', 84),
        ('Switzerland',             'Algeria',                  '2026-07-02', 85),
        ('Argentina',               'Cape Verde',               '2026-07-03', 86),
        ('Colombia',                'Ghana',                    '2026-07-03', 87),
        ('Australia',               'Egypt',                    '2026-07-03', 88),
    ]

    print("\n" + "=" * 90)
    print(f"{'R32':<4} {'Home':<25} {'Pred':^7} {'Away':<25} "
          f"{'λH':>5} {'λA':>5}  {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'MostLikelyProb':>14}")
    print("=" * 90)

    for home, away, date, fifa_no in R32_FIXTURES:
        res = predict_match_score(home, away, lambda_cache,
                                  cls_proba_cache=cls_proba_cache,
                                  cls_blend=args.cls_blend)
        score_str = f"{res['pred_a']}-{res['pred_b']}"
        print(f" M{fifa_no:<3} {home:<25} {score_str:^7} {away:<25} "
              f"{res['lambda_a']:>5.2f} {res['lambda_b']:>5.2f}  "
              f"{res['p_win_a']:>5.1f}% {res['p_draw']:>5.1f}% {res['p_win_b']:>5.1f}%  "
              f"{res['pred_score_prob']:>5.1f}%")
        records.append({
            'group': f'R32_M{fifa_no}', 'home_team': home, 'away_team': away,
            'pred_home_score': res['pred_a'],
            'pred_away_score': res['pred_b'],
            'pred_score_str':  score_str,
            'pred_score_prob': res['pred_score_prob'],
            'lambda_home': res['lambda_a'],
            'lambda_away': res['lambda_b'],
            'p_home_win': res['p_win_a'],
            'p_draw':     res['p_draw'],
            'p_away_win': res['p_win_b'],
        })

    print("\n" + "=" * 90)

    R16_FIXTURES = [
        # (home, away, date, FIFA_match_no)
        ('Canada',         'Morocco',   '2026-07-06', 89),
        ('Paraguay',       'France',    '2026-07-06', 90),
        ('Brazil',         'Norway',    '2026-07-07', 91),
        ('Mexico',         'England',   '2026-07-07', 92),
        ('Portugal',       'Spain',     '2026-07-08', 93),
        ('United States',  'Belgium',   '2026-07-08', 94),
        ('Argentina',      'Egypt',     '2026-07-09', 95),
        ('Switzerland',    'Colombia',  '2026-07-09', 96),
    ]

    print("\n" + "=" * 90)
    print(f"{'R16':<4} {'Home':<25} {'Pred':^7} {'Away':<25} "
          f"{'λH':>5} {'λA':>5}  {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'MostLikelyProb':>14}")
    print("=" * 90)

    for home, away, date, fifa_no in R16_FIXTURES:
        res = predict_match_score(home, away, lambda_cache,
                                  cls_proba_cache=cls_proba_cache,
                                  cls_blend=args.cls_blend)
        score_str = f"{res['pred_a']}-{res['pred_b']}"
        print(f" M{fifa_no:<3} {home:<25} {score_str:^7} {away:<25} "
              f"{res['lambda_a']:>5.2f} {res['lambda_b']:>5.2f}  "
              f"{res['p_win_a']:>5.1f}% {res['p_draw']:>5.1f}% {res['p_win_b']:>5.1f}%  "
              f"{res['pred_score_prob']:>5.1f}%")
        records.append({
            'group': f'R16_M{fifa_no}', 'home_team': home, 'away_team': away,
            'pred_home_score': res['pred_a'],
            'pred_away_score': res['pred_b'],
            'pred_score_str':  score_str,
            'pred_score_prob': res['pred_score_prob'],
            'lambda_home': res['lambda_a'],
            'lambda_away': res['lambda_b'],
            'p_home_win': res['p_win_a'],
            'p_draw':     res['p_draw'],
            'p_away_win': res['p_win_b'],
        })

    QF_FIXTURES = [
        # (home, away, date, FIFA_match_no)
        ('France',      'Morocco',     '2026-07-10', 97),
        ('Spain',       'Belgium',     '2026-07-11', 98),
        ('Norway',      'England',     '2026-07-12', 99),
        ('Argentina',   'Switzerland', '2026-07-12', 100),
    ]

    print("\n" + "=" * 90)
    print(f"{'QF':<4} {'Home':<25} {'Pred':^7} {'Away':<25} "
          f"{'λH':>5} {'λA':>5}  {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'MostLikelyProb':>14}")
    print("=" * 90)

    for home, away, date, fifa_no in QF_FIXTURES:
        res = predict_match_score(home, away, lambda_cache,
                                  cls_proba_cache=cls_proba_cache,
                                  cls_blend=args.cls_blend)
        score_str = f"{res['pred_a']}-{res['pred_b']}"
        print(f" M{fifa_no:<3} {home:<25} {score_str:^7} {away:<25} "
              f"{res['lambda_a']:>5.2f} {res['lambda_b']:>5.2f}  "
              f"{res['p_win_a']:>5.1f}% {res['p_draw']:>5.1f}% {res['p_win_b']:>5.1f}%  "
              f"{res['pred_score_prob']:>5.1f}%")
        records.append({
            'group': f'QF_M{fifa_no}', 'home_team': home, 'away_team': away,
            'pred_home_score': res['pred_a'],
            'pred_away_score': res['pred_b'],
            'pred_score_str':  score_str,
            'pred_score_prob': res['pred_score_prob'],
            'lambda_home': res['lambda_a'],
            'lambda_away': res['lambda_b'],
            'p_home_win': res['p_win_a'],
            'p_draw':     res['p_draw'],
            'p_away_win': res['p_win_b'],
        })

    print("\n" + "=" * 90)

    SF_FIXTURES = [
        # (home, away, date, FIFA_match_no)
        ('France',   'Spain',     '2026-07-15', 101),
        ('England',  'Argentina', '2026-07-16', 102),
    ]

    print("\n" + "=" * 90)
    print(f"{'SF':<4} {'Home':<25} {'Pred':^7} {'Away':<25} "
          f"{'λH':>5} {'λA':>5}  {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'MostLikelyProb':>14}")
    print("=" * 90)

    for home, away, date, fifa_no in SF_FIXTURES:
        res = predict_match_score(home, away, lambda_cache,
                                  cls_proba_cache=cls_proba_cache,
                                  cls_blend=args.cls_blend)
        score_str = f"{res['pred_a']}-{res['pred_b']}"
        print(f" M{fifa_no:<3} {home:<25} {score_str:^7} {away:<25} "
              f"{res['lambda_a']:>5.2f} {res['lambda_b']:>5.2f}  "
              f"{res['p_win_a']:>5.1f}% {res['p_draw']:>5.1f}% {res['p_win_b']:>5.1f}%  "
              f"{res['pred_score_prob']:>5.1f}%")
        records.append({
            'group': f'SF_M{fifa_no}', 'home_team': home, 'away_team': away,
            'pred_home_score': res['pred_a'],
            'pred_away_score': res['pred_b'],
            'pred_score_str':  score_str,
            'pred_score_prob': res['pred_score_prob'],
            'lambda_home': res['lambda_a'],
            'lambda_away': res['lambda_b'],
            'p_home_win': res['p_win_a'],
            'p_draw':     res['p_draw'],
            'p_away_win': res['p_win_b'],
        })

    print("\n" + "=" * 90)

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)
    group_count = df_out[~df_out['group'].str.startswith(('R32', 'R16', 'QF', 'SF'))].shape[0]
    r32_count = df_out[df_out['group'].str.startswith('R32')].shape[0]
    r16_count = df_out[df_out['group'].str.startswith('R16')].shape[0]
    qf_count  = df_out[df_out['group'].str.startswith('QF')].shape[0]
    sf_count  = df_out[df_out['group'].str.startswith('SF')].shape[0]
    print(f"\nSaved {group_count} group stage + {r32_count} R32 + {r16_count} R16 + {qf_count} QF + {sf_count} SF predictions to: {output_path}")

    # === 日本代表の所属グループのみ詳細表示 ===
    japan_group = next((g for g, ts in GROUPS_2026.items() if 'Japan' in ts), None)
    print("\n" + "=" * 60)
    print(f"🇯🇵  JAPAN Group {japan_group} - Detailed Predictions")
    print("=" * 60)
    jp = df_out[df_out['group'] == japan_group].copy()
    for _, row in jp.iterrows():
        print(f"  {row['home_team']:<20} vs {row['away_team']:<20}")
        print(f"    Prediction : {row['pred_score_str']}  "
              f"(probability: {row['pred_score_prob']}%)")
        print(f"    Expected   : {row['lambda_home']:.2f} - {row['lambda_away']:.2f}")
        print(f"    Win/Draw   : {row['p_home_win']}% / {row['p_draw']}% / {row['p_away_win']}%")
        print()


if __name__ == "__main__":
    main()
