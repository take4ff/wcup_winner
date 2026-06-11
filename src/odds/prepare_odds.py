import os
import pandas as pd
import requests
import io

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_path = os.path.join(base_dir, "data/raw/odds/odds_qatar2022.csv")
    
    # 1. グループステージのオッズをダウンロード
    url = "https://raw.githubusercontent.com/stefan-stein/world-cup-2022/main/First%20round%20odds.csv"
    print(f"Downloading group stage odds from {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        df_group = pd.read_csv(io.StringIO(response.text))
        # カラム名の整理: Group,Home,Away,Home win,Draw,Away win
        df_group = df_group.rename(columns={
            'Home': 'home_team',
            'Away': 'away_team',
            'Home win': 'odds_home',
            'Draw': 'odds_draw',
            'Away win': 'odds_away'
        })
        # 不要なカラム削除
        df_group = df_group[['home_team', 'away_team', 'odds_home', 'odds_draw', 'odds_away']]
    else:
        print("Failed to download group stage odds. Initializing empty DataFrame.")
        df_group = pd.DataFrame(columns=['home_team', 'away_team', 'odds_home', 'odds_draw', 'odds_away'])

    # 2. 決勝トーナメント（ノックアウトステージ）のオッズを手動定義（16試合）
    # 対戦表記は results.csv に一致させる
    knockout_data = [
        # ラウンド16
        {"home_team": "Netherlands", "away_team": "United States", "odds_home": 1.95, "odds_draw": 3.30, "odds_away": 4.20},
        {"home_team": "Argentina", "away_team": "Australia", "odds_home": 1.20, "odds_draw": 6.50, "odds_away": 15.00},
        {"home_team": "France", "away_team": "Poland", "odds_home": 1.30, "odds_draw": 5.50, "odds_away": 11.00},
        {"home_team": "England", "away_team": "Senegal", "odds_home": 1.53, "odds_draw": 3.80, "odds_away": 7.50},
        {"home_team": "Japan", "away_team": "Croatia", "odds_home": 3.80, "odds_draw": 3.25, "odds_away": 2.05},
        {"home_team": "Brazil", "away_team": "South Korea", "odds_home": 1.25, "odds_draw": 6.00, "odds_away": 12.00},
        {"home_team": "Morocco", "away_team": "Spain", "odds_home": 6.50, "odds_draw": 4.00, "odds_away": 1.57},
        {"home_team": "Portugal", "away_team": "Switzerland", "odds_home": 1.90, "odds_draw": 3.40, "odds_away": 4.33},
        # 準々決勝
        {"home_team": "Croatia", "away_team": "Brazil", "odds_home": 9.00, "odds_draw": 4.75, "odds_away": 1.36},
        {"home_team": "Netherlands", "away_team": "Argentina", "odds_home": 3.60, "odds_draw": 3.10, "odds_away": 2.20},
        {"home_team": "Morocco", "away_team": "Portugal", "odds_home": 5.50, "odds_draw": 3.60, "odds_away": 1.66},
        {"home_team": "England", "away_team": "France", "odds_home": 3.00, "odds_draw": 3.20, "odds_away": 2.45},
        # 準決勝
        {"home_team": "Argentina", "away_team": "Croatia", "odds_home": 1.83, "odds_draw": 3.40, "odds_away": 4.75},
        {"home_team": "France", "away_team": "Morocco", "odds_home": 1.53, "odds_draw": 3.90, "odds_away": 7.00},
        # 3位決定戦
        {"home_team": "Croatia", "away_team": "Morocco", "odds_home": 2.37, "odds_draw": 3.40, "odds_away": 3.00},
        # 決勝
        {"home_team": "Argentina", "away_team": "France", "odds_home": 2.70, "odds_draw": 3.00, "odds_away": 2.80}
    ]
    
    df_knockout = pd.DataFrame(knockout_data)
    
    # 3. 結合
    df_all = pd.concat([df_group, df_knockout], ignore_index=True)
    
    # チーム名表記の標準化 (results.csvに合わせる)
    team_mapping = {
        'USA': 'United States',
        'South Korea': 'South Korea',
        'Korea Republic': 'South Korea',
    }
    
    df_all['home_team'] = df_all['home_team'].replace(team_mapping)
    df_all['away_team'] = df_all['away_team'].replace(team_mapping)
    
    print(f"Saving combined odds dataset (total {len(df_all)} matches) to {output_path}...")
    df_all.to_csv(output_path, index=False)
    print("Odds preparation completed successfully.")

if __name__ == "__main__":
    main()
