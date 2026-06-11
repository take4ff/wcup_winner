import os
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
from scipy.stats import poisson


# グループとチームの定義 (2022 FIFA World Cup)
GROUPS = {
    'A': ['Qatar', 'Ecuador', 'Senegal', 'Netherlands'],
    'B': ['England', 'Iran', 'United States', 'Wales'],
    'C': ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland'],
    'D': ['France', 'Australia', 'Denmark', 'Tunisia'],
    'E': ['Spain', 'Costa Rica', 'Germany', 'Japan'],
    'F': ['Belgium', 'Canada', 'Morocco', 'Croatia'],
    'G': ['Brazil', 'Serbia', 'Switzerland', 'Cameroon'],
    'H': ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
}

# 主要国の所属連盟（Confederation）定義（開催国カタール＝AFC との一致判定用）
CONFEDERATIONS = {
    'France': 'UEFA', 'Germany': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 
    'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Denmark': 'UEFA', 'Netherlands': 'UEFA',
    'Switzerland': 'UEFA', 'Portugal': 'UEFA', 'Poland': 'UEFA', 'Serbia': 'UEFA',
    'Wales': 'UEFA', 'Italy': 'UEFA', 'Sweden': 'UEFA', 'Ukraine': 'UEFA',
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC',
    'Senegal': 'CAF', 'Morocco': 'CAF', 'Tunisia': 'CAF', 'Cameroon': 'CAF', 'Ghana': 'CAF',
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF', 'Costa Rica': 'CONCACAF'
}

def get_confederation(country_name):
    return CONFEDERATIONS.get(country_name, 'Other')

def load_team_features(features_df):
    team_features = {}
    all_teams = []
    for teams in GROUPS.values():
        all_teams.extend(teams)
        
    # 主力選手離脱ペナルティの定義 (backtest.py と同一)
    SQUAD_VALUE_PENALTIES_2022 = {
        'Senegal': 0.80,
        'France': 0.85,
        'Germany': 0.90,
        'Brazil': 0.95
    }
        
    for team in all_teams:
        # 開幕前（2022-11-20以前）のチームのデータ
        team_matches = features_df[
            ((features_df['home_team'] == team) | (features_df['away_team'] == team)) &
            (features_df['date'] < '2022-11-20')
        ].sort_values(by='date', ascending=False)
        
        if len(team_matches) > 0:
            last_match = team_matches.iloc[0]
            is_home = last_match['home_team'] == team
            prefix = 'home_' if is_home else 'away_'
            
            elo = last_match[f'{prefix}elo_after']
            
            # 離脱ペナルティを適用した選手市場価値
            squad_value = last_match[f'{prefix}squad_value']
            if team in SQUAD_VALUE_PENALTIES_2022:
                squad_value *= SQUAD_VALUE_PENALTIES_2022[team]
            
            team_features[team] = {
                'elo': elo,
                'squad_value': squad_value,
                'last_wcup_matches': last_match[f'{prefix}last_wcup_matches'],
                'conf': get_confederation(team),
                
                # 基本 rolling
                'goals_roll5': last_match[f'{prefix}goals_roll5'],
                'conceded_roll5': last_match[f'{prefix}conceded_roll5'],
                'win_rate_roll5': last_match[f'{prefix}win_rate_roll5'],
                'goals_roll10': last_match[f'{prefix}goals_roll10'],
                'conceded_roll10': last_match[f'{prefix}conceded_roll10'],
                'win_rate_roll10': last_match[f'{prefix}win_rate_roll10'],
                'goals_weighted_roll5': last_match[f'{prefix}goals_weighted_roll5'],
                'conceded_weighted_roll5': last_match[f'{prefix}conceded_weighted_roll5'],
                'goals_weighted_roll10': last_match[f'{prefix}goals_weighted_roll10'],
                'conceded_weighted_roll10': last_match[f'{prefix}conceded_weighted_roll10'],
                
                # 基本 ewm
                'goals_ewm5': last_match[f'{prefix}goals_ewm5'],
                'conceded_ewm5': last_match[f'{prefix}conceded_ewm5'],
                'win_rate_ewm5': last_match[f'{prefix}win_rate_ewm5'],
                'goals_ewm10': last_match[f'{prefix}goals_ewm10'],
                'conceded_ewm10': last_match[f'{prefix}conceded_ewm10'],
                'win_rate_ewm10': last_match[f'{prefix}win_rate_ewm10'],
                'goals_weighted_ewm5': last_match[f'{prefix}goals_weighted_ewm5'],
                'conceded_weighted_ewm5': last_match[f'{prefix}conceded_weighted_ewm5'],
                'goals_weighted_ewm10': last_match[f'{prefix}goals_weighted_ewm10'],
                'conceded_weighted_ewm10': last_match[f'{prefix}conceded_weighted_ewm10'],
                
                # 公式戦優先 rolling
                'goals_official_roll5': last_match[f'{prefix}goals_official_roll5'],
                'conceded_official_roll5': last_match[f'{prefix}conceded_official_roll5'],
                'win_rate_official_roll5': last_match[f'{prefix}win_rate_official_roll5'],
                'goals_official_roll10': last_match[f'{prefix}goals_official_roll10'],
                'conceded_official_roll10': last_match[f'{prefix}conceded_official_roll10'],
                'win_rate_official_roll10': last_match[f'{prefix}win_rate_official_roll10'],
                'goals_weighted_official_roll5': last_match[f'{prefix}goals_weighted_official_roll5'],
                'conceded_weighted_official_roll5': last_match[f'{prefix}conceded_weighted_official_roll5'],
                'goals_weighted_official_roll10': last_match[f'{prefix}goals_weighted_official_roll10'],
                'conceded_weighted_official_roll10': last_match[f'{prefix}conceded_weighted_official_roll10'],

                # 公式戦優先 ewm
                'goals_official_ewm5': last_match[f'{prefix}goals_official_ewm5'],
                'conceded_official_ewm5': last_match[f'{prefix}conceded_official_ewm5'],
                'win_rate_official_ewm5': last_match[f'{prefix}win_rate_official_ewm5'],
                'goals_official_ewm10': last_match[f'{prefix}goals_official_ewm10'],
                'conceded_official_ewm10': last_match[f'{prefix}conceded_official_ewm10'],
                'win_rate_official_ewm10': last_match[f'{prefix}win_rate_official_ewm10'],
                'goals_weighted_official_ewm5': last_match[f'{prefix}goals_weighted_official_ewm5'],
                'conceded_weighted_official_ewm5': last_match[f'{prefix}conceded_weighted_official_ewm5'],
                'goals_weighted_official_ewm10': last_match[f'{prefix}goals_weighted_official_ewm10'],
                'conceded_weighted_official_ewm10': last_match[f'{prefix}conceded_weighted_official_ewm10'],
            }
        else:
            # デフォルト値
            team_features[team] = {
                'elo': 1500.0,
                'squad_value': 150.0,
                'last_wcup_matches': 0,
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
    
    same_conf_a = 1 if feat_a['conf'] == 'AFC' else 0
    same_conf_b = 1 if feat_b['conf'] == 'AFC' else 0
    
    is_host_a = 1 if team_a == 'Qatar' else 0
    is_host_b = 1 if team_b == 'Qatar' else 0
    
    return {
        'elo_diff': feat_a['elo'] - feat_b['elo'],
        'squad_value_diff': feat_a['squad_value'] - feat_b['squad_value'],
        'last_wcup_matches_own': feat_a['last_wcup_matches'],
        'last_wcup_matches_opp': feat_b['last_wcup_matches'],
        
        # 実質ホームアドバンテージ
        'same_conf_own': same_conf_a,
        'same_conf_opp': same_conf_b,
        'is_host_own': is_host_a,
        'is_host_opp': is_host_b,
        
        # 基本 rolling
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

        # 基本 ewm
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
        
        # 公式戦優先 rolling
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

        # 公式戦優先 ewm
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
        
        'was_home': 0
    }

def precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols):
    all_teams = list(team_feats.keys())
    pairs = []
    rows = []
    
    for t1 in all_teams:
        for t2 in all_teams:
            if t1 != t2:
                pairs.append((t1, t2))
                rows.append(make_match_features(t1, t2, team_feats))
                
    X = pd.DataFrame(rows)[feature_cols]
    
    lambdas_poisson = model_poisson.predict(X)
    lambdas_lgbm = model_lgbm.predict(X)
    lambdas = (lambdas_poisson + lambdas_lgbm) / 2.0
    
    lambda_cache = {}
    for (t1, t2), lam in zip(pairs, lambdas):
        lambda_cache[(t1, t2)] = lam
        
    return lambda_cache

def predict_match(team_a, team_b, lambda_cache, rho=-0.03, max_goals=10):
    lambda_a = lambda_cache[(team_a, team_b)]
    lambda_b = lambda_cache[(team_b, team_a)]
    
    goals = np.arange(max_goals + 1)
    h_probs = poisson.pmf(goals, lambda_a)
    a_probs = poisson.pmf(goals, lambda_b)
    
    prob_matrix = np.outer(h_probs, a_probs)
    prob_matrix /= prob_matrix.sum()
    
    if abs(rho) > 1e-9:
        tau_00 = 1.0 - lambda_a * lambda_b * rho
        tau_10 = 1.0 + lambda_b * rho
        tau_01 = 1.0 + lambda_a * rho
        tau_11 = 1.0 - rho
        
        prob_matrix[0, 0] *= max(tau_00, 0.0)
        prob_matrix[1, 0] *= max(tau_10, 0.0)
        prob_matrix[0, 1] *= max(tau_01, 0.0)
        prob_matrix[1, 1] *= max(tau_11, 0.0)
        
        prob_matrix /= prob_matrix.sum()
        
    flat_probs = prob_matrix.flatten()
    sampled_idx = np.random.choice(len(flat_probs), p=flat_probs)
    
    score_a = sampled_idx // (max_goals + 1)
    score_b = sampled_idx % (max_goals + 1)
    
    return score_a, score_b


def simulate_group_stage(team_feats, lambda_cache):
    group_winners = {}
    group_runners_up = {}
    
    for grp, teams in GROUPS.items():
        pts = defaultdict(int)
        gf = defaultdict(int)
        ga = defaultdict(int)
        
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                s1, s2 = predict_match(t1, t2, lambda_cache)
                
                gf[t1] += s1
                ga[t1] += s2
                gf[t2] += s2
                ga[t2] += s1
                
                if s1 > s2:
                    pts[t1] += 3
                elif s1 < s2:
                    pts[t2] += 3
                else:
                    pts[t1] += 1
                    pts[t2] += 1
                    
        standings = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t]),
            reverse=True
        )
        
        group_winners[grp] = standings[0]
        group_runners_up[grp] = standings[1]
        
    return group_winners, group_runners_up

def simulate_knockout_match(t1, t2, lambda_cache):
    s1, s2 = predict_match(t1, t2, lambda_cache)
    if s1 > s2:
        return t1, t2
    elif s1 < s2:
        return t2, t1
    else:
        if np.random.rand() < 0.5:
            return t1, t2
        else:
            return t2, t1

def simulate_tournament(team_feats, lambda_cache):
    winners, runners_up = simulate_group_stage(team_feats, lambda_cache)
    progress = {}
    
    r16_matches = [
        (winners['A'], runners_up['B']),
        (winners['C'], runners_up['D']),
        (winners['D'], runners_up['C']),
        (winners['B'], runners_up['A']),
        (winners['E'], runners_up['F']),
        (winners['G'], runners_up['H']),
        (winners['F'], runners_up['E']),
        (winners['H'], runners_up['G'])
    ]
    
    for t in team_feats.keys():
        progress[t] = 'GroupStage'
        
    for t in list(winners.values()) + list(runners_up.values()):
        progress[t] = 'R16'
        
    w_qf = []
    for t1, t2 in r16_matches:
        win, loss = simulate_knockout_match(t1, t2, lambda_cache)
        w_qf.append(win)
        progress[win] = 'QF'
        
    qf_matches = [
        (w_qf[0], w_qf[1]),
        (w_qf[4], w_qf[5]),
        (w_qf[2], w_qf[3]),
        (w_qf[6], w_qf[7])
    ]
    
    w_sf = []
    for t1, t2 in qf_matches:
        win, loss = simulate_knockout_match(t1, t2, lambda_cache)
        w_sf.append(win)
        progress[win] = 'SF'
        
    sf_matches = [
        (w_sf[0], w_sf[1]),
        (w_sf[2], w_sf[3])
    ]
    
    w_final = []
    l_final = []
    for t1, t2 in sf_matches:
        win, loss = simulate_knockout_match(t1, t2, lambda_cache)
        w_final.append(win)
        l_final.append(loss)
        
    progress[w_final[0]] = 'Final'
    progress[w_final[1]] = 'Final'
    
    third_place_win, third_place_loss = simulate_knockout_match(l_final[0], l_final[1], lambda_cache)
    progress[third_place_win] = '3rd'
    
    champion, runner_up = simulate_knockout_match(w_final[0], w_final[1], lambda_cache)
    progress[champion] = 'Champion'
    progress[runner_up] = '2nd'
    
    return progress

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path = os.path.join(base_dir, "data/processed/features.csv")
    poisson_model_path = os.path.join(base_dir, "models/poisson_model.joblib")
    lgbm_model_path = os.path.join(base_dir, "models/lgbm_model.joblib")
    feature_cols_path = os.path.join(base_dir, "models/feature_cols.joblib")
    
    print("Loading models and features...")
    df_features = pd.read_csv(features_path)
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    model_poisson = joblib.load(poisson_model_path)
    model_lgbm = joblib.load(lgbm_model_path)
    feature_cols = joblib.load(feature_cols_path)
    
    print("Caching team features just before the World Cup...")
    team_feats = load_team_features(df_features)
    
    print("Precomputing expected goals (lambdas) for all possible matchups...")
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols)
    
    num_simulations = 10000
    print(f"Running Monte Carlo Simulation ({num_simulations} iterations)...")
    
    stage_counts = defaultdict(lambda: defaultdict(int))
    
    for sim in range(num_simulations):
        if (sim + 1) % 2000 == 0:
            print(f"  Completed {sim + 1} simulations...")
            
        progress = simulate_tournament(team_feats, lambda_cache)
        for team, stage in progress.items():
            stage_counts[team][stage] += 1
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
        counts = stage_counts[team]
        rows.append({
            'team': team,
            'Group_Stage_Exit_Prob': (counts['GroupStage'] / num_simulations) * 100,
            'R16_Prob': (counts['Reached_R16'] / num_simulations) * 100,
            'QF_Prob': (counts['Reached_QF'] / num_simulations) * 100,
            'SF_Prob': (counts['Reached_SF'] / num_simulations) * 100,
            'Final_Prob': (counts['Reached_Final'] / num_simulations) * 100,
            'Winner_Prob': (counts['Champion_Count'] / num_simulations) * 100
        })
        
    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values(by='Winner_Prob', ascending=False).reset_index(drop=True)
    
    print("\n================ TOP 10 SIMULATION RESULTS (WINNER PROBABILITY) ================")
    print(df_results.head(10).to_string(index=False))
    print("================================================================================\n")
    
    target_teams = ['Japan', 'Brazil', 'Argentina', 'France', 'England', 'Germany', 'Spain']
    print("================ SPECIFIC TEAMS PROBABILITY ================")
    print(df_results[df_results['team'].isin(target_teams)].to_string(index=False))
    print("============================================================\n")
    
    output_path = os.path.join(base_dir, "data/processed/simulation_results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Simulation results saved to {output_path}")

if __name__ == "__main__":
    main()
