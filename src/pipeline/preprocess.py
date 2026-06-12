import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from altitude import city_altitude, team_home_altitude  # noqa: E402

# 所属連盟（Confederation）定義
# 全FIFA加盟国 + 歴史的代表（消滅国）+ 連盟管轄の準加盟地域をカバーする。
# CONIFA等の非公式代表（Padania, Sealand など）は意図的に 'Other' のまま。
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
    'Albania': 'UEFA', 'Andorra': 'UEFA', 'Armenia': 'UEFA', 'Azerbaijan': 'UEFA',
    'Belarus': 'UEFA', 'Bulgaria': 'UEFA', 'Cyprus': 'UEFA', 'Estonia': 'UEFA',
    'Faroe Islands': 'UEFA', 'Georgia': 'UEFA', 'Gibraltar': 'UEFA', 'Israel': 'UEFA',
    'Kazakhstan': 'UEFA', 'Kosovo': 'UEFA', 'Latvia': 'UEFA', 'Liechtenstein': 'UEFA',
    'Lithuania': 'UEFA', 'Luxembourg': 'UEFA', 'Malta': 'UEFA', 'Moldova': 'UEFA',
    'Montenegro': 'UEFA', 'North Macedonia': 'UEFA', 'San Marino': 'UEFA', 'Slovenia': 'UEFA',
    # UEFA: 歴史的代表
    'Czechoslovakia': 'UEFA', 'Yugoslavia': 'UEFA', 'FR Yugoslavia': 'UEFA',
    'Serbia and Montenegro': 'UEFA', 'German DR': 'UEFA', 'Soviet Union': 'UEFA',
    'Ireland': 'UEFA', 'Irish Free State': 'UEFA', 'Éire': 'UEFA',
    'Bohemia': 'UEFA', 'Bohemia and Moravia': 'UEFA', 'Saarland': 'UEFA',
    # CONMEBOL (南米)
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    # AFC (アジア)
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC', 'China': 'AFC', 'Iraq': 'AFC', 'UAE': 'AFC',
    'Uzbekistan': 'AFC', 'Oman': 'AFC', 'Jordan': 'AFC', 'Syria': 'AFC', 'Vietnam': 'AFC',
    'Afghanistan': 'AFC', 'Bahrain': 'AFC', 'Bangladesh': 'AFC', 'Bhutan': 'AFC',
    'Brunei': 'AFC', 'Cambodia': 'AFC', 'Hong Kong': 'AFC', 'India': 'AFC',
    'Indonesia': 'AFC', 'Kuwait': 'AFC', 'Kyrgyzstan': 'AFC', 'Laos': 'AFC',
    'Lebanon': 'AFC', 'Macau': 'AFC', 'Malaysia': 'AFC', 'Maldives': 'AFC',
    'Mongolia': 'AFC', 'Myanmar': 'AFC', 'Nepal': 'AFC', 'North Korea': 'AFC',
    'Pakistan': 'AFC', 'Palestine': 'AFC', 'Philippines': 'AFC', 'Singapore': 'AFC',
    'Sri Lanka': 'AFC', 'Taiwan': 'AFC', 'Tajikistan': 'AFC', 'Thailand': 'AFC',
    'Timor-Leste': 'AFC', 'Turkmenistan': 'AFC', 'United Arab Emirates': 'AFC',
    'Yemen': 'AFC', 'Guam': 'AFC', 'Northern Mariana Islands': 'AFC',
    # AFC: 歴史的代表
    'Burma': 'AFC', 'Ceylon': 'AFC', 'Malaya': 'AFC', 'North Vietnam': 'AFC',
    'Vietnam Republic': 'AFC', 'Yemen AR': 'AFC', 'Yemen DPR': 'AFC', 'South Yemen': 'AFC',
    # CAF (アフリカ)
    'Senegal': 'CAF', 'Morocco': 'CAF', 'Tunisia': 'CAF', 'Cameroon': 'CAF',
    'Ghana': 'CAF', 'Egypt': 'CAF', 'Nigeria': 'CAF', 'Algeria': 'CAF',
    'Ivory Coast': 'CAF', 'South Africa': 'CAF', 'Mali': 'CAF', 'Burkina Faso': 'CAF',
    'DR Congo': 'CAF', 'Guinea': 'CAF', 'Angola': 'CAF',
    'Benin': 'CAF', 'Botswana': 'CAF', 'Burundi': 'CAF', 'Cape Verde': 'CAF',
    'Central African Republic': 'CAF', 'Chad': 'CAF', 'Comoros': 'CAF', 'Congo': 'CAF',
    'Djibouti': 'CAF', 'Equatorial Guinea': 'CAF', 'Eritrea': 'CAF', 'Eswatini': 'CAF',
    'Ethiopia': 'CAF', 'Gabon': 'CAF', 'Gambia': 'CAF', 'Guinea-Bissau': 'CAF',
    'Kenya': 'CAF', 'Lesotho': 'CAF', 'Liberia': 'CAF', 'Libya': 'CAF',
    'Madagascar': 'CAF', 'Malawi': 'CAF', 'Mauritania': 'CAF', 'Mauritius': 'CAF',
    'Mozambique': 'CAF', 'Namibia': 'CAF', 'Niger': 'CAF', 'Rwanda': 'CAF',
    'Seychelles': 'CAF', 'Sierra Leone': 'CAF', 'Somalia': 'CAF', 'South Sudan': 'CAF',
    'Sudan': 'CAF', 'São Tomé and Príncipe': 'CAF', 'Tanzania': 'CAF', 'Togo': 'CAF',
    'Uganda': 'CAF', 'Zambia': 'CAF', 'Zimbabwe': 'CAF',
    'Réunion': 'CAF', 'Zanzibar': 'CAF',
    # CAF: 歴史的代表
    'Belgian Congo': 'CAF', 'Congo-Kinshasa': 'CAF', 'Zaïre': 'CAF', 'Dahomey': 'CAF',
    'French Somaliland': 'CAF', 'Gold Coast': 'CAF', 'Northern Rhodesia': 'CAF',
    'Southern Rhodesia': 'CAF', 'Nyasaland': 'CAF', 'Swaziland': 'CAF',
    'Tanganyika': 'CAF', 'Upper Volta': 'CAF', 'United Arab Republic': 'CAF',
    # CONCACAF (北中米・カリブ海)
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Honduras': 'CONCACAF', 'Jamaica': 'CONCACAF', 'Panama': 'CONCACAF', 'El Salvador': 'CONCACAF',
    'Trinidad and Tobago': 'CONCACAF',
    'Anguilla': 'CONCACAF', 'Antigua and Barbuda': 'CONCACAF', 'Aruba': 'CONCACAF',
    'Bahamas': 'CONCACAF', 'Barbados': 'CONCACAF', 'Belize': 'CONCACAF',
    'Bermuda': 'CONCACAF', 'Bonaire': 'CONCACAF', 'British Virgin Islands': 'CONCACAF',
    'Cayman Islands': 'CONCACAF', 'Cuba': 'CONCACAF', 'Curaçao': 'CONCACAF',
    'Dominica': 'CONCACAF', 'Dominican Republic': 'CONCACAF', 'French Guiana': 'CONCACAF',
    'Grenada': 'CONCACAF', 'Guadeloupe': 'CONCACAF', 'Guatemala': 'CONCACAF',
    'Guyana': 'CONCACAF', 'Haiti': 'CONCACAF', 'Martinique': 'CONCACAF',
    'Montserrat': 'CONCACAF', 'Nicaragua': 'CONCACAF', 'Puerto Rico': 'CONCACAF',
    'Saint Kitts and Nevis': 'CONCACAF', 'Saint Lucia': 'CONCACAF', 'Saint Martin': 'CONCACAF',
    'Saint Vincent and the Grenadines': 'CONCACAF', 'Sint Maarten': 'CONCACAF',
    'Suriname': 'CONCACAF', 'Turks and Caicos Islands': 'CONCACAF',
    'United States Virgin Islands': 'CONCACAF',
    # CONCACAF: 歴史的代表
    'British Guiana': 'CONCACAF', 'Dutch Guyana': 'CONCACAF',
    # OFC (オセアニア)
    'New Zealand': 'OFC', 'Fiji': 'OFC', 'New Caledonia': 'OFC', 'Papua New Guinea': 'OFC',
    'Samoa': 'OFC', 'American Samoa': 'OFC', 'Solomon Islands': 'OFC', 'Tahiti': 'OFC',
    'Tonga': 'OFC', 'Vanuatu': 'OFC', 'Cook Islands': 'OFC', 'Kiribati': 'OFC',
    'Niue': 'OFC', 'Tuvalu': 'OFC',
    # OFC: 歴史的代表
    'New Hebrides': 'OFC', 'Western Samoa': 'OFC',
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
    wcup_years = [2006, 2010, 2014, 2018, 2022]
    wcup_end_dates = {
        2006: pd.to_datetime('2006-07-09'),
        2010: pd.to_datetime('2010-07-11'),
        2014: pd.to_datetime('2014-07-13'),
        2018: pd.to_datetime('2018-07-15'),
        2022: pd.to_datetime('2022-12-18'),
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
    
    # 選手市場価値（Squad Value）: 3スナップショットの日付線形補間
    # 旧実装は「2022-12-18以降は一律2026年値」だったため、2023年の試合に未来の
    # 市場価値が混入するリークがあった。スナップショット日付間で線形補間する。
    print("Merging Squad Value features (date-interpolated)...")
    snapshots = [
        (pd.Timestamp('2018-06-01'), pd.read_csv(os.path.join(base_dir, "data/raw/squad/squad_values_2018.csv"))),
        (pd.Timestamp('2022-11-01'), pd.read_csv(os.path.join(base_dir, "data/raw/squad/squad_values_2022.csv"))),
        (pd.Timestamp('2026-06-01'), pd.read_csv(os.path.join(base_dir, "data/raw/squad/squad_values_2026.csv"))),
    ]
    # チーム -> (スナップショット日付epoch[], 値[])
    team_series = {}
    for snap_date, snap_df in snapshots:
        for r in snap_df.itertuples():
            team_series.setdefault(r.team, ([], []))
            team_series[r.team][0].append(snap_date.value)
            team_series[r.team][1].append(float(r.squad_value_eur_million))

    def interp_value(team, date_val):
        s = team_series.get(team)
        if s is None:
            return np.nan
        # 範囲外は端の値で埋める（np.interpのデフォルト挙動）
        return float(np.interp(date_val, s[0], s[1]))

    date_vals = df_features['date'].astype('int64')
    df_features['home_squad_value'] = [
        interp_value(t, d) for t, d in zip(df_features['home_team'], date_vals)]
    df_features['away_squad_value'] = [
        interp_value(t, d) for t, d in zip(df_features['away_team'], date_vals)]

    # 市場価値の欠損フラグ（モデルに「データなし」を明示する）と控えめなデフォルト値
    # ※ 市場価値ファイルは主にW杯出場国を収録しているため、欠損＝中堅以下の国が大半。
    #    一律150M€は弱小国を過大評価するため 50M€ に変更。
    df_features['home_squad_value_missing'] = df_features['home_squad_value'].isna().astype(int)
    df_features['away_squad_value_missing'] = df_features['away_squad_value'].isna().astype(int)
    df_features['home_squad_value'] = df_features['home_squad_value'].fillna(50.0)
    df_features['away_squad_value'] = df_features['away_squad_value'].fillna(50.0)

    # 休養日数（前試合からの経過日数, 上限30日, 初出場は30日扱い）
    print("Adding rest-day features...")
    appearances = pd.concat([
        df_features[['date', 'home_team']].rename(columns={'home_team': 'team'}),
        df_features[['date', 'away_team']].rename(columns={'away_team': 'team'}),
    ]).drop_duplicates().sort_values(['team', 'date'])
    appearances['rest_days'] = appearances.groupby('team')['date'].diff().dt.days
    appearances['rest_days'] = appearances['rest_days'].clip(upper=30).fillna(30.0)

    df_features = df_features.merge(
        appearances.rename(columns={'team': 'home_team', 'rest_days': 'home_rest_days'}),
        on=['date', 'home_team'], how='left')
    df_features = df_features.merge(
        appearances.rename(columns={'team': 'away_team', 'rest_days': 'away_rest_days'}),
        on=['date', 'away_team'], how='left')
    df_features['home_rest_days'] = df_features['home_rest_days'].fillna(30.0)
    df_features['away_rest_days'] = df_features['away_rest_days'].fillna(30.0)

    # 標高差特徴量（開催都市の標高 - チーム本拠の標高。高地未順応の不利を表現）
    print("Adding altitude features...")
    venue_alt = df_features['city'].map(city_altitude)
    df_features['home_altitude_diff'] = venue_alt - df_features['home_team'].map(team_home_altitude)
    df_features['away_altitude_diff'] = venue_alt - df_features['away_team'].map(team_home_altitude)
    
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
