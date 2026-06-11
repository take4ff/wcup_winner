"""
predict_scores_2026.py
2026年FIFAワールドカップ グループステージ全48試合 + ノックアウト想定試合の
最確スコア・期待得点・勝敗確率を計算してCSV出力・コンソール表示する。
"""
import os
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator_2026 import (
    GROUPS_2026, load_team_features, make_match_features, precompute_all_lambdas
)


RHO = -0.03    # Dixon-Coles 補正パラメータ (backtest最適値)
MAX_GOALS = 10


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


def predict_match_score(team_a, team_b, lambda_cache, rho=RHO):
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

    return {
        'pred_a': int(pred_a), 'pred_b': int(pred_b),
        'pred_score_prob': round(pred_prob * 100, 1),
        'lambda_a': round(la, 3), 'lambda_b': round(lb, 3),
        'p_win_a': round(p_win_a * 100, 1),
        'p_draw':  round(p_draw  * 100, 1),
        'p_win_b': round(p_win_b * 100, 1),
    }


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path     = os.path.join(base_dir, "data/processed/features.csv")
    poisson_model_path = os.path.join(base_dir, "models/poisson_model.joblib")
    lgbm_model_path   = os.path.join(base_dir, "models/lgbm_model.joblib")
    feature_cols_path  = os.path.join(base_dir, "models/feature_cols.joblib")
    output_path        = os.path.join(base_dir, "data/processed/2026/predicted_scores.csv")

    print("Loading models and features...")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    model_poisson = joblib.load(poisson_model_path)
    model_lgbm    = joblib.load(lgbm_model_path)
    feature_cols  = joblib.load(feature_cols_path)

    print("Loading team features...")
    team_feats = load_team_features(df)

    print("Precomputing lambdas...")
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols)

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
                res = predict_match_score(home, away, lambda_cache)
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

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved all {len(df_out)} match predictions to: {output_path}")

    # === 日本代表グループKのみ詳細表示 ===
    print("\n" + "=" * 60)
    print("🇯🇵  JAPAN Group K - Detailed Predictions")
    print("=" * 60)
    jp = df_out[df_out['group'] == 'K'].copy()
    for _, row in jp.iterrows():
        print(f"  {row['home_team']:<20} vs {row['away_team']:<20}")
        print(f"    Prediction : {row['pred_score_str']}  "
              f"(probability: {row['pred_score_prob']}%)")
        print(f"    Expected   : {row['lambda_home']:.2f} - {row['lambda_away']:.2f}")
        print(f"    Win/Draw   : {row['p_home_win']}% / {row['p_draw']}% / {row['p_away_win']}%")
        print()


if __name__ == "__main__":
    main()
