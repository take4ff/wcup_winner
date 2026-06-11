"""
simulator_2026.py - 2026年FIFAワールドカップ（48チーム）モンテカルロ・シミュレーター

大会フォーマット:
  グループステージ: 12グループ × 4チーム... ではなく、正式には 12グループ × 4チーム
  ※ 2026年は当初3チーム/グループ案から 4チーム/12グループ に変更済み
  - 12グループ (A〜L)、各4チーム、総試合数 48
  - 各グループ上位2チーム (24チーム) + 8つのベスト3位 = 32チーム → R32進出
  - 以降: R32 → QF(8) → SF(4) → Final
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
from scipy.stats import poisson

# ===================================================================
# 2026年 FIFA World Cup グループ定義 (12グループ × 4チーム)
# ===================================================================
GROUPS_2026 = {
    # Group A: Mexico, South Africa, Korea Republic (→South Korea), Czech Republic
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    # Group B: Canada, Bosnia and Herzegovina, Qatar, Switzerland
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    # Group C: Brazil, Morocco, Haiti, Scotland
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    # Group D: United States, Paraguay, Australia, Turkey
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    # Group E: Germany, Curaçao, Ivory Coast, Ecuador
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    # Group F: Netherlands, Japan, Tunisia, Sweden (UEFAプレーオフB優勝)
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    # Group G: Belgium, Egypt, Iran, New Zealand
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    # Group H: Spain, Cape Verde, Saudi Arabia, Uruguay
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    # Group I: France, Senegal, Norway, Iraq (大陸間プレーオフ2優勝)
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    # Group J: Argentina, Algeria, Austria, Jordan
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    # Group K: Portugal, Uzbekistan, Colombia, DR Congo (大陸間プレーオフ1優勝)
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    # Group L: England, Croatia, Ghana, Panama
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

CONFEDERATIONS_2026 = {
    # UEFA
    'France': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Belgium': 'UEFA', 'Italy': 'UEFA',
    'Croatia': 'UEFA', 'Denmark': 'UEFA', 'Switzerland': 'UEFA', 'Serbia': 'UEFA',
    'Poland': 'UEFA', 'Romania': 'UEFA', 'Austria': 'UEFA', 'Slovakia': 'UEFA',
    'Czech Republic': 'UEFA', 'Scotland': 'UEFA', 'Turkey': 'UEFA', 'Georgia': 'UEFA',
    'Albania': 'UEFA', 'Slovenia': 'UEFA', 'Hungary': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Norway': 'UEFA',
    # CONMEBOL
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Bolivia': 'CONMEBOL', 'Venezuela': 'CONMEBOL',
    # AFC
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC', 'Uzbekistan': 'AFC', 'Jordan': 'AFC',
    # CAF
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Egypt': 'CAF', 'Nigeria': 'CAF',
    'Cameroon': 'CAF', 'Mali': 'CAF', 'South Africa': 'CAF', 'Ivory Coast': 'CAF',
    'Algeria': 'CAF', 'Tunisia': 'CAF', 'Ghana': 'CAF', 'Cape Verde': 'CAF',
    'DR Congo': 'CAF',
    'Haiti': 'CONCACAF',
    # CONCACAF
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Panama': 'CONCACAF', 'Honduras': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Jamaica': 'CONCACAF', 'Curaçao': 'CONCACAF',
    # AFC
    'Iraq': 'AFC',
    # UEFA extra
    'Sweden': 'UEFA',
    # OFC
    'New Zealand': 'OFC',
}

# 開催国
HOST_COUNTRIES = {'United States', 'Canada', 'Mexico'}


def get_confederation(team):
    return CONFEDERATIONS_2026.get(team, 'Other')


def load_team_features(features_df):
    """2026年W杯開幕前の最新チーム特徴量を取得"""
    team_features = {}
    all_teams = []
    for teams in GROUPS_2026.values():
        all_teams.extend(teams)

    for team in all_teams:
        team_matches = features_df[
            ((features_df['home_team'] == team) | (features_df['away_team'] == team)) &
            (features_df['date'] < '2026-06-11')
        ].sort_values(by='date', ascending=False)

        if len(team_matches) > 0:
            last = team_matches.iloc[0]
            is_home = last['home_team'] == team
            prefix = 'home_' if is_home else 'away_'

            elo = last[f'{prefix}elo_after']
            squad_value = last[f'{prefix}squad_value']

            def safe_get(col):
                return last[col] if col in last.index else 1.2

            team_features[team] = {
                'elo': elo,
                'squad_value': squad_value,
                'last_wcup_matches': last[f'{prefix}last_wcup_matches'],
                'conf': get_confederation(team),
                'goals_roll5':   safe_get(f'{prefix}goals_roll5'),
                'conceded_roll5': safe_get(f'{prefix}conceded_roll5'),
                'win_rate_roll5': safe_get(f'{prefix}win_rate_roll5'),
                'goals_roll10':  safe_get(f'{prefix}goals_roll10'),
                'conceded_roll10': safe_get(f'{prefix}conceded_roll10'),
                'win_rate_roll10': safe_get(f'{prefix}win_rate_roll10'),
                'goals_weighted_roll5':  safe_get(f'{prefix}goals_weighted_roll5'),
                'conceded_weighted_roll5': safe_get(f'{prefix}conceded_weighted_roll5'),
                'goals_weighted_roll10': safe_get(f'{prefix}goals_weighted_roll10'),
                'conceded_weighted_roll10': safe_get(f'{prefix}conceded_weighted_roll10'),
                'goals_ewm5':   safe_get(f'{prefix}goals_ewm5'),
                'conceded_ewm5': safe_get(f'{prefix}conceded_ewm5'),
                'win_rate_ewm5': safe_get(f'{prefix}win_rate_ewm5'),
                'goals_ewm10':  safe_get(f'{prefix}goals_ewm10'),
                'conceded_ewm10': safe_get(f'{prefix}conceded_ewm10'),
                'win_rate_ewm10': safe_get(f'{prefix}win_rate_ewm10'),
                'goals_weighted_ewm5':  safe_get(f'{prefix}goals_weighted_ewm5'),
                'conceded_weighted_ewm5': safe_get(f'{prefix}conceded_weighted_ewm5'),
                'goals_weighted_ewm10': safe_get(f'{prefix}goals_weighted_ewm10'),
                'conceded_weighted_ewm10': safe_get(f'{prefix}conceded_weighted_ewm10'),
                'goals_official_roll5':  safe_get(f'{prefix}goals_official_roll5'),
                'conceded_official_roll5': safe_get(f'{prefix}conceded_official_roll5'),
                'win_rate_official_roll5': safe_get(f'{prefix}win_rate_official_roll5'),
                'goals_official_roll10': safe_get(f'{prefix}goals_official_roll10'),
                'conceded_official_roll10': safe_get(f'{prefix}conceded_official_roll10'),
                'win_rate_official_roll10': safe_get(f'{prefix}win_rate_official_roll10'),
                'goals_weighted_official_roll5': safe_get(f'{prefix}goals_weighted_official_roll5'),
                'conceded_weighted_official_roll5': safe_get(f'{prefix}conceded_weighted_official_roll5'),
                'goals_weighted_official_roll10': safe_get(f'{prefix}goals_weighted_official_roll10'),
                'conceded_weighted_official_roll10': safe_get(f'{prefix}conceded_weighted_official_roll10'),
                'goals_official_ewm5':  safe_get(f'{prefix}goals_official_ewm5'),
                'conceded_official_ewm5': safe_get(f'{prefix}conceded_official_ewm5'),
                'win_rate_official_ewm5': safe_get(f'{prefix}win_rate_official_ewm5'),
                'goals_official_ewm10': safe_get(f'{prefix}goals_official_ewm10'),
                'conceded_official_ewm10': safe_get(f'{prefix}conceded_official_ewm10'),
                'win_rate_official_ewm10': safe_get(f'{prefix}win_rate_official_ewm10'),
                'goals_weighted_official_ewm5': safe_get(f'{prefix}goals_weighted_official_ewm5'),
                'conceded_weighted_official_ewm5': safe_get(f'{prefix}conceded_weighted_official_ewm5'),
                'goals_weighted_official_ewm10': safe_get(f'{prefix}goals_weighted_official_ewm10'),
                'conceded_weighted_official_ewm10': safe_get(f'{prefix}conceded_weighted_official_ewm10'),
            }
        else:
            print(f"  [WARNING] No data found for {team}, using default values.")
            team_features[team] = {
                'elo': 1500.0, 'squad_value': 150.0, 'last_wcup_matches': 0,
                'conf': get_confederation(team),
                'goals_roll5': 1.2, 'conceded_roll5': 1.2, 'win_rate_roll5': 0.5,
                'goals_roll10': 1.2, 'conceded_roll10': 1.2, 'win_rate_roll10': 0.5,
                'goals_weighted_roll5': 1.2, 'conceded_weighted_roll5': 1.2,
                'goals_weighted_roll10': 1.2, 'conceded_weighted_roll10': 1.2,
                'goals_ewm5': 1.2, 'conceded_ewm5': 1.2, 'win_rate_ewm5': 0.5,
                'goals_ewm10': 1.2, 'conceded_ewm10': 1.2, 'win_rate_ewm10': 0.5,
                'goals_weighted_ewm5': 1.2, 'conceded_weighted_ewm5': 1.2,
                'goals_weighted_ewm10': 1.2, 'conceded_weighted_ewm10': 1.2,
                'goals_official_roll5': 1.2, 'conceded_official_roll5': 1.2, 'win_rate_official_roll5': 0.5,
                'goals_official_roll10': 1.2, 'conceded_official_roll10': 1.2, 'win_rate_official_roll10': 0.5,
                'goals_weighted_official_roll5': 1.2, 'conceded_weighted_official_roll5': 1.2,
                'goals_weighted_official_roll10': 1.2, 'conceded_weighted_official_roll10': 1.2,
                'goals_official_ewm5': 1.2, 'conceded_official_ewm5': 1.2, 'win_rate_official_ewm5': 0.5,
                'goals_official_ewm10': 1.2, 'conceded_official_ewm10': 1.2, 'win_rate_official_ewm10': 0.5,
                'goals_weighted_official_ewm5': 1.2, 'conceded_weighted_official_ewm5': 1.2,
                'goals_weighted_official_ewm10': 1.2, 'conceded_weighted_official_ewm10': 1.2,
            }

    return team_features


def make_match_features(team_a, team_b, team_feats):
    feat_a = team_feats[team_a]
    feat_b = team_feats[team_b]

    same_conf_a = 1 if feat_a['conf'] == 'CONCACAF' else 0
    same_conf_b = 1 if feat_b['conf'] == 'CONCACAF' else 0
    is_host_a = 1 if team_a in HOST_COUNTRIES else 0
    is_host_b = 1 if team_b in HOST_COUNTRIES else 0

    return {
        'elo_diff': feat_a['elo'] - feat_b['elo'],
        'squad_value_diff': feat_a['squad_value'] - feat_b['squad_value'],
        'last_wcup_matches_own': feat_a['last_wcup_matches'],
        'last_wcup_matches_opp': feat_b['last_wcup_matches'],
        'same_conf_own': same_conf_a, 'same_conf_opp': same_conf_b,
        'is_host_own': is_host_a, 'is_host_opp': is_host_b,
        'goals_roll5_own': feat_a['goals_roll5'],
        'conceded_roll5_own': feat_a['conceded_roll5'],
        'win_rate_roll5_own': feat_a['win_rate_roll5'],
        'goals_roll10_own': feat_a['goals_roll10'],
        'conceded_roll10_own': feat_a['conceded_roll10'],
        'win_rate_roll10_own': feat_a['win_rate_roll10'],
        'goals_weighted_roll5_own': feat_a['goals_weighted_roll5'],
        'conceded_weighted_roll5_own': feat_a['conceded_weighted_roll5'],
        'goals_weighted_roll10_own': feat_a['goals_weighted_roll10'],
        'conceded_weighted_roll10_own': feat_a['conceded_weighted_roll10'],
        'goals_roll5_opp': feat_b['goals_roll5'],
        'conceded_roll5_opp': feat_b['conceded_roll5'],
        'win_rate_roll5_opp': feat_b['win_rate_roll5'],
        'goals_roll10_opp': feat_b['goals_roll10'],
        'conceded_roll10_opp': feat_b['conceded_roll10'],
        'win_rate_roll10_opp': feat_b['win_rate_roll10'],
        'goals_weighted_roll5_opp': feat_b['goals_weighted_roll5'],
        'conceded_weighted_roll5_opp': feat_b['conceded_weighted_roll5'],
        'goals_weighted_roll10_opp': feat_b['goals_weighted_roll10'],
        'conceded_weighted_roll10_opp': feat_b['conceded_weighted_roll10'],
        'goals_ewm5_own': feat_a['goals_ewm5'],
        'conceded_ewm5_own': feat_a['conceded_ewm5'],
        'win_rate_ewm5_own': feat_a['win_rate_ewm5'],
        'goals_ewm10_own': feat_a['goals_ewm10'],
        'conceded_ewm10_own': feat_a['conceded_ewm10'],
        'win_rate_ewm10_own': feat_a['win_rate_ewm10'],
        'goals_weighted_ewm5_own': feat_a['goals_weighted_ewm5'],
        'conceded_weighted_ewm5_own': feat_a['conceded_weighted_ewm5'],
        'goals_weighted_ewm10_own': feat_a['goals_weighted_ewm10'],
        'conceded_weighted_ewm10_own': feat_a['conceded_weighted_ewm10'],
        'goals_ewm5_opp': feat_b['goals_ewm5'],
        'conceded_ewm5_opp': feat_b['conceded_ewm5'],
        'win_rate_ewm5_opp': feat_b['win_rate_ewm5'],
        'goals_ewm10_opp': feat_b['goals_ewm10'],
        'conceded_ewm10_opp': feat_b['conceded_ewm10'],
        'win_rate_ewm10_opp': feat_b['win_rate_ewm10'],
        'goals_weighted_ewm5_opp': feat_b['goals_weighted_ewm5'],
        'conceded_weighted_ewm5_opp': feat_b['conceded_weighted_ewm5'],
        'goals_weighted_ewm10_opp': feat_b['goals_weighted_ewm10'],
        'conceded_weighted_ewm10_opp': feat_b['conceded_weighted_ewm10'],
        'goals_official_roll5_own': feat_a['goals_official_roll5'],
        'conceded_official_roll5_own': feat_a['conceded_official_roll5'],
        'win_rate_official_roll5_own': feat_a['win_rate_official_roll5'],
        'goals_official_roll10_own': feat_a['goals_official_roll10'],
        'conceded_official_roll10_own': feat_a['conceded_official_roll10'],
        'win_rate_official_roll10_own': feat_a['win_rate_official_roll10'],
        'goals_weighted_official_roll5_own': feat_a['goals_weighted_official_roll5'],
        'conceded_weighted_official_roll5_own': feat_a['conceded_weighted_official_roll5'],
        'goals_weighted_official_roll10_own': feat_a['goals_weighted_official_roll10'],
        'conceded_weighted_official_roll10_own': feat_a['conceded_weighted_official_roll10'],
        'goals_official_roll5_opp': feat_b['goals_official_roll5'],
        'conceded_official_roll5_opp': feat_b['conceded_official_roll5'],
        'win_rate_official_roll5_opp': feat_b['win_rate_official_roll5'],
        'goals_official_roll10_opp': feat_b['goals_official_roll10'],
        'conceded_official_roll10_opp': feat_b['conceded_official_roll10'],
        'win_rate_official_roll10_opp': feat_b['win_rate_official_roll10'],
        'goals_weighted_official_roll5_opp': feat_b['goals_weighted_official_roll5'],
        'conceded_weighted_official_roll5_opp': feat_b['conceded_weighted_official_roll5'],
        'goals_weighted_official_roll10_opp': feat_b['goals_weighted_official_roll10'],
        'conceded_weighted_official_roll10_opp': feat_b['conceded_weighted_official_roll10'],
        'goals_official_ewm5_own': feat_a['goals_official_ewm5'],
        'conceded_official_ewm5_own': feat_a['conceded_official_ewm5'],
        'win_rate_official_ewm5_own': feat_a['win_rate_official_ewm5'],
        'goals_official_ewm10_own': feat_a['goals_official_ewm10'],
        'conceded_official_ewm10_own': feat_a['conceded_official_ewm10'],
        'win_rate_official_ewm10_own': feat_a['win_rate_official_ewm10'],
        'goals_weighted_official_ewm5_own': feat_a['goals_weighted_official_ewm5'],
        'conceded_weighted_official_ewm5_own': feat_a['conceded_weighted_official_ewm5'],
        'goals_weighted_official_ewm10_own': feat_a['goals_weighted_official_ewm10'],
        'conceded_weighted_official_ewm10_own': feat_a['conceded_weighted_official_ewm10'],
        'goals_official_ewm5_opp': feat_b['goals_official_ewm5'],
        'conceded_official_ewm5_opp': feat_b['conceded_official_ewm5'],
        'win_rate_official_ewm5_opp': feat_b['win_rate_official_ewm5'],
        'goals_official_ewm10_opp': feat_b['goals_official_ewm10'],
        'conceded_official_ewm10_opp': feat_b['conceded_official_ewm10'],
        'win_rate_official_ewm10_opp': feat_b['win_rate_official_ewm10'],
        'goals_weighted_official_ewm5_opp': feat_b['goals_weighted_official_ewm5'],
        'conceded_weighted_official_ewm5_opp': feat_b['conceded_weighted_official_ewm5'],
        'goals_weighted_official_ewm10_opp': feat_b['goals_weighted_official_ewm10'],
        'conceded_weighted_official_ewm10_opp': feat_b['conceded_weighted_official_ewm10'],
        'was_home': 0,
    }


def precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols):
    all_teams = list(team_feats.keys())
    pairs, rows = [], []
    for t1 in all_teams:
        for t2 in all_teams:
            if t1 != t2:
                pairs.append((t1, t2))
                rows.append(make_match_features(t1, t2, team_feats))

    X = pd.DataFrame(rows)[feature_cols]
    lambdas = (model_poisson.predict(X) + model_lgbm.predict(X)) / 2.0

    lambda_cache = {}
    for (t1, t2), lam in zip(pairs, lambdas):
        lambda_cache[(t1, t2)] = lam
    return lambda_cache


def predict_match(team_a, team_b, lambda_cache, rho=-0.03, max_goals=10):
    lambda_a = lambda_cache[(team_a, team_b)]
    lambda_b = lambda_cache[(team_b, team_a)]
    goals = np.arange(max_goals + 1)
    prob_matrix = np.outer(poisson.pmf(goals, lambda_a), poisson.pmf(goals, lambda_b))
    prob_matrix /= prob_matrix.sum()
    if abs(rho) > 1e-9:
        prob_matrix[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 0.0)
        prob_matrix[1, 0] *= max(1.0 + lambda_b * rho, 0.0)
        prob_matrix[0, 1] *= max(1.0 + lambda_a * rho, 0.0)
        prob_matrix[1, 1] *= max(1.0 - rho, 0.0)
        prob_matrix /= prob_matrix.sum()
    flat = prob_matrix.flatten()
    idx = np.random.choice(len(flat), p=flat)
    return idx // (max_goals + 1), idx % (max_goals + 1)


def simulate_group_stage(lambda_cache):
    """グループステージ: 12グループ × 4チーム、各組上位2チーム確定 + 全3位チームを返す"""
    group_top2 = {}   # grp -> [1位, 2位]
    all_third = []    # (grp, チーム名, pts, gd, gf)

    for grp, teams in GROUPS_2026.items():
        pts = defaultdict(int)
        gf  = defaultdict(int)
        ga  = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                s1, s2 = predict_match(t1, t2, lambda_cache)
                gf[t1] += s1; ga[t1] += s2
                gf[t2] += s2; ga[t2] += s1
                if s1 > s2: pts[t1] += 3
                elif s1 < s2: pts[t2] += 3
                else: pts[t1] += 1; pts[t2] += 1

        standings = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t]),
            reverse=True
        )
        group_top2[grp] = standings[:2]
        third = standings[2]
        all_third.append((grp, third, pts[third], gf[third] - ga[third], gf[third]))

    # ベスト3位 8チームを選出 (pts→GD→GF順)
    all_third_sorted = sorted(all_third, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [t[1] for t in all_third_sorted[:8]]

    return group_top2, best_thirds


def simulate_knockout_match(t1, t2, lambda_cache):
    s1, s2 = predict_match(t1, t2, lambda_cache)
    if s1 > s2: return t1, t2
    elif s1 < s2: return t2, t1
    else:
        return (t1, t2) if np.random.rand() < 0.5 else (t2, t1)


def simulate_tournament(lambda_cache):
    group_top2, best_thirds = simulate_group_stage(lambda_cache)
    progress = {}

    # グループ落ちチームを記録
    for grp, teams in GROUPS_2026.items():
        for t in teams:
            progress[t] = 'GroupStage'

    # R32進出チーム
    r32_teams = []
    for grp in sorted(GROUPS_2026.keys()):
        for t in group_top2[grp]:
            progress[t] = 'R32'
            r32_teams.append(t)
    for t in best_thirds:
        progress[t] = 'R32'
        r32_teams.append(t)

    # R32 (32→16)
    np.random.shuffle(r32_teams)
    r16_teams = []
    for i in range(0, 32, 2):
        win, _ = simulate_knockout_match(r32_teams[i], r32_teams[i+1], lambda_cache)
        progress[win] = 'R16'
        r16_teams.append(win)

    # QF (16→8)
    qf_teams = []
    for i in range(0, 16, 2):
        win, _ = simulate_knockout_match(r16_teams[i], r16_teams[i+1], lambda_cache)
        progress[win] = 'QF'
        qf_teams.append(win)

    # SF (8→4)
    sf_teams = []
    for i in range(0, 8, 2):
        win, _ = simulate_knockout_match(qf_teams[i], qf_teams[i+1], lambda_cache)
        progress[win] = 'SF'
        sf_teams.append(win)

    # 準決勝 (4→2)
    finalists, losers = [], []
    for i in range(0, 4, 2):
        win, loss = simulate_knockout_match(sf_teams[i], sf_teams[i+1], lambda_cache)
        progress[win] = 'Final'
        finalists.append(win)
        losers.append(loss)

    # 3位決定戦
    third, _ = simulate_knockout_match(losers[0], losers[1], lambda_cache)
    progress[third] = '3rd'

    # 決勝
    champion, runner_up = simulate_knockout_match(finalists[0], finalists[1], lambda_cache)
    progress[champion] = 'Champion'
    progress[runner_up] = '2nd'

    return progress


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path        = os.path.join(base_dir, "data/processed/features.csv")
    poisson_model_path   = os.path.join(base_dir, "models/poisson_model.joblib")
    lgbm_model_path      = os.path.join(base_dir, "models/lgbm_model.joblib")
    feature_cols_path    = os.path.join(base_dir, "models/feature_cols.joblib")
    output_path          = os.path.join(base_dir, "data/processed/2026/simulation_results.csv")

    print("Loading models and features...")
    df_features = pd.read_csv(features_path)
    df_features['date'] = pd.to_datetime(df_features['date'])

    model_poisson  = joblib.load(poisson_model_path)
    model_lgbm     = joblib.load(lgbm_model_path)
    feature_cols   = joblib.load(feature_cols_path)

    print("Caching team features just before the 2026 World Cup...")
    team_feats = load_team_features(df_features)
    print(f"  {len(team_feats)} teams loaded.")

    print("Precomputing expected goals (lambdas) for all possible matchups...")
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols)

    num_simulations = 10000
    print(f"Running Monte Carlo Simulation ({num_simulations} iterations)...\n")

    stage_counts = defaultdict(lambda: defaultdict(int))

    for sim in range(num_simulations):
        if (sim + 1) % 2000 == 0:
            print(f"  Completed {sim + 1} simulations...")

        progress = simulate_tournament(lambda_cache)
        for team, stage in progress.items():
            stage_counts[team][stage] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF', 'R16', 'R32']:
                stage_counts[team]['Reached_R32'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF', 'R16']:
                stage_counts[team]['Reached_R16'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF']:
                stage_counts[team]['Reached_QF'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF']:
                stage_counts[team]['Reached_SF'] += 1
            if stage in ['Champion', '2nd', 'Final']:
                stage_counts[team]['Reached_Final'] += 1
            if stage == 'Champion':
                stage_counts[team]['Champion_Count'] += 1

    rows = []
    for team in team_feats.keys():
        elo_val = round(team_feats[team]['elo'], 1)
        c = stage_counts[team]
        grp = next((g for g, ts in GROUPS_2026.items() if team in ts), '?')
        rows.append({
            'team': team, 'group': grp, 'elo': elo_val,
            'R32_Prob':    round((c['Reached_R32']   / num_simulations) * 100, 2),
            'R16_Prob':    round((c['Reached_R16']   / num_simulations) * 100, 2),
            'QF_Prob':     round((c['Reached_QF']    / num_simulations) * 100, 2),
            'SF_Prob':     round((c['Reached_SF']    / num_simulations) * 100, 2),
            'Final_Prob':  round((c['Reached_Final'] / num_simulations) * 100, 2),
            'Winner_Prob': round((c['Champion_Count'] / num_simulations) * 100, 2),
        })

    df_results = pd.DataFrame(rows).sort_values(by='Winner_Prob', ascending=False).reset_index(drop=True)

    print("\n================ TOP 15 SIMULATION RESULTS (WINNER PROBABILITY) ================")
    print(df_results.head(15).to_string(index=False))
    print("=================================================================================\n")

    focus_teams = ['Japan', 'Brazil', 'Argentina', 'France', 'England', 'Germany',
                   'Spain', 'United States', 'Canada', 'Mexico', 'Morocco', 'Portugal']
    print("================ SPECIFIC TEAMS PROBABILITY ================")
    print(df_results[df_results['team'].isin(focus_teams)].to_string(index=False))
    print("============================================================\n")

    df_results.to_csv(output_path, index=False)
    print(f"Simulation results saved to {output_path}")


if __name__ == "__main__":
    main()
