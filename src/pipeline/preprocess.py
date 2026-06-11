import os
import pandas as pd
import numpy as np

# 主要国の所属連盟（Confederation）定義
CONFEDERATIONS = {
    # UEFA (ヨーロッパ)
    'France': 'UEFA', 'Germany': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 
    'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Denmark': 'UEFA', 'Netherlands': 'UEFA',
    'Switzerland': 'UEFA', 'Portugal': 'UEFA', 'Poland': 'UEFA', 'Serbia': 'UEFA',
    'Wales': 'UEFA', 'Italy': 'UEFA', 'Sweden': 'UEFA', 'Ukraine': 'UEFA',
    'Russia': 'UEFA', 'Scotland': 'UEFA', 'Turkey': 'UEFA', 'Austria': 'UEFA',
    'Czech Republic': 'UEFA', 'Hungary': 'UEFA', 'Slovakia': 'UEFA', 'Romania': 'UEFA',
    'Norway': 'UEFA', 'Greece': 'UEFA', 'Republic of Ireland': 'UEFA', 'Northern Ireland': 'UEFA',
    'Finland': 'UEFA', 'Iceland': 'UEFA', 'Bosnia and Herzegovina': 'UEFA',
    # CONMEBOL (南米)
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    # AFC (アジア)
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC', 'China': 'AFC', 'Iraq': 'AFC', 'UAE': 'AFC',
    'Uzbekistan': 'AFC', 'Oman': 'AFC', 'Jordan': 'AFC', 'Syria': 'AFC', 'Vietnam': 'AFC',
    # CAF (アフリカ)
    'Senegal': 'CAF', 'Morocco': 'CAF', 'Tunisia': 'CAF', 'Cameroon': 'CAF',
    'Ghana': 'CAF', 'Egypt': 'CAF', 'Nigeria': 'CAF', 'Algeria': 'CAF',
    'Ivory Coast': 'CAF', 'South Africa': 'CAF', 'Mali': 'CAF', 'Burkina Faso': 'CAF',
    'DR Congo': 'CAF', 'Guinea': 'CAF', 'Angola': 'CAF',
    # CONCACAF (北中米・カリブ海)
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Honduras': 'CONCACAF', 'Jamaica': 'CONCACAF', 'Panama': 'CONCACAF', 'El Salvador': 'CONCACAF',
    'Trinidad and Tobago': 'CONCACAF'
}

def get_confederation(country_name):
    return CONFEDERATIONS.get(country_name, 'Other')

def create_features():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_path = os.path.join(base_dir, "data/processed/results_with_elo.csv")
    squad_value_path = os.path.join(base_dir, "data/raw/squad/squad_values_2022.csv")
    output_path = os.path.join(base_dir, "data/processed/features.csv")
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 直近のW杯の終了日の定義と、過去大会の各チーム試合数の集計
    wcup_years = [2006, 2010, 2014, 2018]
    wcup_end_dates = {
        2006: pd.to_datetime('2006-07-09'),
        2010: pd.to_datetime('2010-07-11'),
        2014: pd.to_datetime('2014-07-13'),
        2018: pd.to_datetime('2018-07-15'),
    }
    wcup_stats = {}
    for year in wcup_years:
        wcup_matches = df[
            (df['tournament'] == 'FIFA World Cup') & 
            (df['date'].dt.year == year)
        ]
        teams = pd.concat([wcup_matches['home_team'], wcup_matches['away_team']])
        wcup_stats[year] = teams.value_counts().to_dict()

    def get_last_wcup_matches(team, m_date):
        last_year = None
        for year in sorted(wcup_years, reverse=True):
            if m_date > wcup_end_dates[year]:
                last_year = year
                break
        if last_year is None:
            return 0
        return wcup_stats[last_year].get(team, 0)

    print("Adding last World Cup match count features...")
    df['home_last_wcup_matches'] = df.apply(lambda r: get_last_wcup_matches(r['home_team'], r['date']), axis=1)
    df['away_last_wcup_matches'] = df.apply(lambda r: get_last_wcup_matches(r['away_team'], r['date']), axis=1)

    # 試合結果の数値化（勝=1, 分=0.5, 負=0）
    def get_result_points(home_score, away_score, team_type):
        if home_score > away_score:
            return 1.0 if team_type == 'home' else 0.0
        elif home_score < away_score:
            return 0.0 if team_type == 'home' else 1.0
        else:
            return 0.5

    print("Expanding match data for rolling features...")
    rows = []
    for idx, row in df.iterrows():
        rows.append({
            'match_idx': idx,
            'date': row['date'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'opponent_elo': row['away_elo_before'],
            'goals': row['home_score'],
            'goals_conceded': row['away_score'],
            'points': get_result_points(row['home_score'], row['away_score'], 'home'),
            'was_home': 1 if not row['neutral'] else 0,
            'tournament': row['tournament']
        })
        rows.append({
            'match_idx': idx,
            'date': row['date'],
            'team': row['away_team'],
            'opponent': row['home_team'],
            'opponent_elo': row['home_elo_before'],
            'goals': row['away_score'],
            'goals_conceded': row['home_score'],
            'points': get_result_points(row['home_score'], row['away_score'], 'away'),
            'was_home': 0,
            'tournament': row['tournament']
        })
        
    expanded_df = pd.DataFrame(rows)
    expanded_df = expanded_df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # 相手Eloで調整した得失点 (Opponent-Adjusted Stats)
    # 相手が強い(Eloが高い)ほど得点の価値を高く、相手が弱いほど失点の価値(やらかし度)を高く補正する
    expanded_df['opponent_elo'] = expanded_df['opponent_elo'].fillna(1500.0)
    
    # 得点の補正（相手のEloが高いほど価値が上がる。0.5倍〜2.0倍に制限）
    adj_factor_goals = 1.0 + 0.5 * (expanded_df['opponent_elo'] - 1500.0) / 500.0
    adj_factor_goals = adj_factor_goals.clip(0.5, 2.0)
    expanded_df['goals_weighted'] = expanded_df['goals'] * adj_factor_goals
    
    # 失点の補正（相手のEloが低い＝弱いほど価値が上がる。0.5倍〜2.0倍に制限）
    adj_factor_conceded = 1.0 + 0.5 * (1500.0 - expanded_df['opponent_elo']) / 500.0
    adj_factor_conceded = adj_factor_conceded.clip(0.5, 2.0)
    expanded_df['conceded_weighted'] = expanded_df['goals_conceded'] * adj_factor_conceded
    
    # 1. 通常の rolling (単純平均 + ewm指数平均) を計算
    print("Calculating general rolling and ewm features...")
    grouped = expanded_df.groupby('team')
    
    for window in [5, 10]:
        # 単純移動平均
        expanded_df[f'goals_roll{window}'] = grouped['goals'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_df[f'conceded_roll{window}'] = grouped['goals_conceded'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_df[f'win_rate_roll{window}'] = grouped['points'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_df[f'goals_weighted_roll{window}'] = grouped['goals_weighted'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_df[f'conceded_weighted_roll{window}'] = grouped['conceded_weighted'].shift(1).rolling(window=window, min_periods=1).mean()
        
        # 指数移動平均 (時間減衰 Time Decay 適用)
        expanded_df[f'goals_ewm{window}'] = grouped['goals'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_df[f'conceded_ewm{window}'] = grouped['goals_conceded'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_df[f'win_rate_ewm{window}'] = grouped['points'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_df[f'goals_weighted_ewm{window}'] = grouped['goals_weighted'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_df[f'conceded_weighted_ewm{window}'] = grouped['conceded_weighted'].shift(1).ewm(span=window, adjust=False).mean()

    # 2. 公式戦のみの rolling (単純平均 + ewm指数平均) を計算
    print("Calculating official match rolling and ewm features...")
    expanded_df['is_official'] = expanded_df['tournament'].apply(lambda x: 0 if 'friendly' in str(x).lower() else 1)
    
    expanded_official = expanded_df[expanded_df['is_official'] == 1].copy()
    grouped_official = expanded_official.groupby('team')
    
    for window in [5, 10]:
        # 単純移動平均
        expanded_official[f'goals_official_roll{window}'] = grouped_official['goals'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_official[f'conceded_official_roll{window}'] = grouped_official['goals_conceded'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_official[f'win_rate_official_roll{window}'] = grouped_official['points'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_official[f'goals_weighted_official_roll{window}'] = grouped_official['goals_weighted'].shift(1).rolling(window=window, min_periods=1).mean()
        expanded_official[f'conceded_weighted_official_roll{window}'] = grouped_official['conceded_weighted'].shift(1).rolling(window=window, min_periods=1).mean()
        
        # 指数移動平均
        expanded_official[f'goals_official_ewm{window}'] = grouped_official['goals'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_official[f'conceded_official_ewm{window}'] = grouped_official['goals_conceded'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_official[f'win_rate_official_ewm{window}'] = grouped_official['points'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_official[f'goals_weighted_official_ewm{window}'] = grouped_official['goals_weighted'].shift(1).ewm(span=window, adjust=False).mean()
        expanded_official[f'conceded_weighted_official_ewm{window}'] = grouped_official['conceded_weighted'].shift(1).ewm(span=window, adjust=False).mean()

    # 公式戦の計算結果を元の expanded_df にマージし、時系列に沿って forward fill
    official_cols = [col for col in expanded_official.columns if 'official_roll' in col or 'official_ewm' in col]
    expanded_df = expanded_df.merge(
        expanded_official[['date', 'team'] + official_cols],
        on=['date', 'team'],
        how='left'
    )
    expanded_df = expanded_df.sort_values(by=['team', 'date']).reset_index(drop=True)
    expanded_df[official_cols] = expanded_df.groupby('team')[official_cols].ffill()

    # 元の試合データフレームにマージして戻す
    print("Merging features back to main DataFrame...")
    
    home_rows = expanded_df.merge(
        df[['home_team']].rename_axis('match_idx').reset_index(),
        left_on=['match_idx', 'team'],
        right_on=['match_idx', 'home_team']
    ).drop(columns=['home_team'])
    
    away_rows = expanded_df.merge(
        df[['away_team']].rename_axis('match_idx').reset_index(),
        left_on=['match_idx', 'team'],
        right_on=['match_idx', 'away_team']
    ).drop(columns=['away_team'])
    
    # すべての時系列特徴量カラムを抽出
    time_series_cols = [col for col in expanded_df.columns if 'roll' in col or 'ewm' in col]
    
    home_rows = home_rows[['match_idx'] + time_series_cols].rename(columns={col: f'home_{col}' for col in time_series_cols})
    away_rows = away_rows[['match_idx'] + time_series_cols].rename(columns={col: f'away_{col}' for col in time_series_cols})
    
    df_features = df.copy()
    df_features['match_idx'] = df_features.index
    
    df_features = df_features.merge(home_rows, on='match_idx', how='left')
    df_features = df_features.merge(away_rows, on='match_idx', how='left')
    
    # 選手市場価値（Squad Value）のロードとマージ（時系列分割マージ）
    print("Merging Squad Value features based on match dates...")
    squad_2018_path = os.path.join(base_dir, "data/raw/squad/squad_values_2018.csv")
    squad_2022_path = os.path.join(base_dir, "data/raw/squad/squad_values_2022.csv")
    squad_2026_path = os.path.join(base_dir, "data/raw/squad/squad_values_2026.csv")
    
    df_squad_2018 = pd.read_csv(squad_2018_path)
    df_squad_2022 = pd.read_csv(squad_2022_path)
    df_squad_2026 = pd.read_csv(squad_2026_path)
    
    # 期間別に分割
    cond_2018 = df_features['date'] <= '2018-07-15'
    cond_2022 = (df_features['date'] > '2018-07-15') & (df_features['date'] <= '2022-12-18')
    cond_2026 = df_features['date'] > '2022-12-18'
    
    df_part_2018 = df_features[cond_2018].copy()
    df_part_2022 = df_features[cond_2022].copy()
    df_part_2026 = df_features[cond_2026].copy()
    
    # マージ用ヘルパー関数
    def merge_squad(df_part, df_squad):
        if len(df_part) == 0:
            df_part['home_squad_value'] = np.nan
            df_part['away_squad_value'] = np.nan
            return df_part
        df_part = df_part.merge(df_squad.rename(columns={'team': 'home_team', 'squad_value_eur_million': 'home_squad_value'}), on='home_team', how='left')
        df_part = df_part.merge(df_squad.rename(columns={'team': 'away_team', 'squad_value_eur_million': 'away_squad_value'}), on='away_team', how='left')
        return df_part
        
    df_part_2018 = merge_squad(df_part_2018, df_squad_2018)
    df_part_2022 = merge_squad(df_part_2022, df_squad_2022)
    df_part_2026 = merge_squad(df_part_2026, df_squad_2026)
    
    # 再結合
    df_features = pd.concat([df_part_2018, df_part_2022, df_part_2026], ignore_index=True)
    df_features = df_features.sort_values(by='date').reset_index(drop=True)
    
    df_features['home_squad_value'] = df_features['home_squad_value'].fillna(150.0)
    df_features['away_squad_value'] = df_features['away_squad_value'].fillna(150.0)
    
    # 実質ホームアドバンテージ（連盟一致・開催国）の追加
    print("Adding confederation and host features...")
    df_features['home_conf'] = df_features['home_team'].apply(get_confederation)
    df_features['away_conf'] = df_features['away_team'].apply(get_confederation)
    df_features['country_conf'] = df_features['country'].apply(get_confederation)
    
    df_features['home_same_confederation'] = (df_features['home_conf'] == df_features['country_conf']).astype(int)
    df_features['away_same_confederation'] = (df_features['away_conf'] == df_features['country_conf']).astype(int)
    
    df_features.loc[df_features['home_conf'] == 'Other', 'home_same_confederation'] = 0
    df_features.loc[df_features['away_conf'] == 'Other', 'away_same_confederation'] = 0
    
    df_features['home_is_host'] = (df_features['home_team'] == df_features['country']).astype(int)
    df_features['away_is_host'] = (df_features['away_team'] == df_features['country']).astype(int)
    
    # その他の特徴量
    df_features['elo_diff'] = df_features['home_elo_before'] - df_features['away_elo_before']
    df_features['squad_value_diff'] = df_features['home_squad_value'] - df_features['away_squad_value']
    df_features['last_wcup_matches_diff'] = df_features['home_last_wcup_matches'] - df_features['away_last_wcup_matches']
    df_features['neutral'] = df_features['neutral'].astype(int)
    
    # 欠損値補完の設計
    fill_values = {}
    for team_prefix in ['home', 'away']:
        for window in [5, 10]:
            for type_suffix in ['', '_official']:
                for mode in ['roll', 'ewm']:
                    fill_values[f'{team_prefix}_goals{type_suffix}_{mode}{window}'] = 1.2
                    fill_values[f'{team_prefix}_conceded{type_suffix}_{mode}{window}'] = 1.2
                    fill_values[f'{team_prefix}_win_rate{type_suffix}_{mode}{window}'] = 0.5
                    fill_values[f'{team_prefix}_goals_weighted{type_suffix}_{mode}{window}'] = 1.2
                    fill_values[f'{team_prefix}_conceded_weighted{type_suffix}_{mode}{window}'] = 1.2
                    
    df_features = df_features.fillna(value=fill_values)
    
    print(f"Saving preprocessed features to {output_path}...")
    df_features.to_csv(output_path, index=False)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    create_features()
