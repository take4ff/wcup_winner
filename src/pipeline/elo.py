import os
import pandas as pd
import numpy as np
from collections import defaultdict

def get_k_factor(tournament):
    # K値は walkforward.py --mode elo のグリッド評価で旧値×1.2が最良だったため調整済み
    # (2000年以降の全試合で Brier 0.13413 → 0.13397)
    tournament = str(tournament).lower()
    if 'world cup' in tournament:
        if 'qualification' in tournament or 'qualifiers' in tournament:
            return 36
        else:
            return 60  # W杯本大会は高く設定
    elif 'euro' in tournament or 'copa' in tournament or 'asian cup' in tournament or 'africa cup' in tournament or 'gold cup' in tournament:
        if 'qualification' in tournament or 'qualifiers' in tournament:
            return 36
        else:
            return 48  # 大陸選手権本大会
    elif 'friendly' in tournament:
        return 12  # 親善試合は低く設定
    else:
        return 24  # その他

def get_goal_difference_multiplier(home_score, away_score):
    d = abs(home_score - away_score)
    if d <= 1:
        return 1.0
    elif d == 2:
        return 1.5
    else:
        return (11.0 + d) / 8.0

def calculate_elo():
    # パス設定
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_path = os.path.join(base_dir, "data/raw/match/results.csv")
    output_path = os.path.join(base_dir, "data/processed/results_with_elo.csv")
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 欠損値処理
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    # 日付でソート
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Eloレーティングの初期化
    elo_dict = defaultdict(lambda: 1500.0)
    
    home_elo_before_list = []
    away_elo_before_list = []
    home_elo_after_list = []
    away_elo_after_list = []
    
    print("Calculating Elo ratings...")
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        tournament = row['tournament']
        neutral = row['neutral']
        
        # 試合前のレーティングを保持
        r_home = elo_dict[home_team]
        r_away = elo_dict[away_team]
        
        home_elo_before_list.append(r_home)
        away_elo_before_list.append(r_away)
        
        # ホームアドバンテージ補正 (中立地でなければ +100)
        h_adv = 0 if neutral else 100
        
        # 期待勝率の計算
        dr = (r_home + h_adv) - r_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        expected_away = 1.0 - expected_home
        
        # 実際の結果
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
            
        # Kファクターと得点差補正の計算
        k = get_k_factor(tournament)
        mult = get_goal_difference_multiplier(home_score, away_score)
        
        # レーティングの更新
        new_r_home = r_home + k * mult * (actual_home - expected_home)
        new_r_away = r_away + k * mult * (actual_away - expected_away)
        
        # 辞書の更新
        elo_dict[home_team] = new_r_home
        elo_dict[away_team] = new_r_away
        
        # 試合後のレーティングを保存
        home_elo_after_list.append(new_r_home)
        away_elo_after_list.append(new_r_away)
        
    df['home_elo_before'] = home_elo_before_list
    df['away_elo_before'] = away_elo_before_list
    df['home_elo_after'] = home_elo_after_list
    df['away_elo_after'] = away_elo_after_list
    
    print(f"Saving results with Elo rating to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Elo calculation completed successfully.")

if __name__ == "__main__":
    calculate_elo()
