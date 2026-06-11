import os
import pandas as pd

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_path = os.path.join(base_dir, "data/raw/odds/odds_russia2018.csv")

    # 2018年 FIFAワールドカップ ロシア大会 全64試合のオッズ（開幕前のブックメーカー平均値）
    # チーム名表記は data/raw/results.csv に合わせる
    odds_data = [
        # ============ グループステージ ============
        # Group A
        {"home_team": "Russia",        "away_team": "Saudi Arabia",  "odds_home": 1.50, "odds_draw": 4.00, "odds_away": 7.50},
        {"home_team": "Egypt",         "away_team": "Uruguay",       "odds_home": 3.80, "odds_draw": 3.20, "odds_away": 2.15},
        {"home_team": "Russia",        "away_team": "Egypt",         "odds_home": 1.85, "odds_draw": 3.40, "odds_away": 4.75},
        {"home_team": "Uruguay",       "away_team": "Saudi Arabia",  "odds_home": 1.38, "odds_draw": 4.75, "odds_away": 9.50},
        {"home_team": "Uruguay",       "away_team": "Russia",        "odds_home": 2.40, "odds_draw": 3.10, "odds_away": 3.10},
        {"home_team": "Saudi Arabia",  "away_team": "Egypt",         "odds_home": 2.40, "odds_draw": 3.10, "odds_away": 3.10},
        # Group B
        {"home_team": "Morocco",       "away_team": "Iran",          "odds_home": 1.87, "odds_draw": 3.40, "odds_away": 4.50},
        {"home_team": "Portugal",      "away_team": "Spain",         "odds_home": 3.10, "odds_draw": 3.10, "odds_away": 2.45},
        {"home_team": "Portugal",      "away_team": "Morocco",       "odds_home": 1.55, "odds_draw": 3.80, "odds_away": 7.00},
        {"home_team": "Iran",          "away_team": "Spain",         "odds_home": 10.00,"odds_draw": 5.50, "odds_away": 1.29},
        {"home_team": "Iran",          "away_team": "Portugal",      "odds_home": 7.00, "odds_draw": 4.25, "odds_away": 1.53},
        {"home_team": "Spain",         "away_team": "Morocco",       "odds_home": 1.40, "odds_draw": 4.50, "odds_away": 9.00},
        # Group C
        {"home_team": "France",        "away_team": "Australia",     "odds_home": 1.38, "odds_draw": 4.75, "odds_away": 9.50},
        {"home_team": "Peru",          "away_team": "Denmark",       "odds_home": 2.50, "odds_draw": 3.00, "odds_away": 3.10},
        {"home_team": "Denmark",       "away_team": "Australia",     "odds_home": 2.15, "odds_draw": 3.10, "odds_away": 3.60},
        {"home_team": "France",        "away_team": "Peru",          "odds_home": 1.55, "odds_draw": 3.70, "odds_away": 7.00},
        {"home_team": "France",        "away_team": "Denmark",       "odds_home": 1.80, "odds_draw": 3.30, "odds_away": 5.00},
        {"home_team": "Australia",     "away_team": "Peru",          "odds_home": 2.50, "odds_draw": 3.10, "odds_away": 3.00},
        # Group D
        {"home_team": "Argentina",     "away_team": "Iceland",       "odds_home": 1.30, "odds_draw": 5.25, "odds_away": 12.00},
        {"home_team": "Croatia",       "away_team": "Nigeria",       "odds_home": 1.85, "odds_draw": 3.40, "odds_away": 4.75},
        {"home_team": "Argentina",     "away_team": "Croatia",       "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 5.00},
        {"home_team": "Nigeria",       "away_team": "Iceland",       "odds_home": 2.20, "odds_draw": 3.10, "odds_away": 3.55},
        {"home_team": "Iceland",       "away_team": "Croatia",       "odds_home": 4.50, "odds_draw": 3.40, "odds_away": 1.87},
        {"home_team": "Nigeria",       "away_team": "Argentina",     "odds_home": 9.50, "odds_draw": 5.00, "odds_away": 1.35},
        # Group E
        {"home_team": "Costa Rica",    "away_team": "Serbia",        "odds_home": 2.60, "odds_draw": 3.10, "odds_away": 2.90},
        {"home_team": "Brazil",        "away_team": "Switzerland",   "odds_home": 1.55, "odds_draw": 3.80, "odds_away": 7.50},
        {"home_team": "Brazil",        "away_team": "Costa Rica",    "odds_home": 1.33, "odds_draw": 5.00, "odds_away": 11.00},
        {"home_team": "Serbia",        "away_team": "Switzerland",   "odds_home": 2.65, "odds_draw": 3.00, "odds_away": 2.85},
        {"home_team": "Brazil",        "away_team": "Serbia",        "odds_home": 1.57, "odds_draw": 3.70, "odds_away": 6.50},
        {"home_team": "Switzerland",   "away_team": "Costa Rica",    "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 4.75},
        # Group F
        {"home_team": "Germany",       "away_team": "Mexico",        "odds_home": 1.53, "odds_draw": 4.00, "odds_away": 7.50},
        {"home_team": "Sweden",        "away_team": "South Korea",   "odds_home": 1.87, "odds_draw": 3.40, "odds_away": 4.50},
        {"home_team": "South Korea",   "away_team": "Mexico",        "odds_home": 5.00, "odds_draw": 3.60, "odds_away": 1.75},
        {"home_team": "Germany",       "away_team": "Sweden",        "odds_home": 1.60, "odds_draw": 3.80, "odds_away": 6.50},
        {"home_team": "South Korea",   "away_team": "Germany",       "odds_home": 9.00, "odds_draw": 5.00, "odds_away": 1.38},
        {"home_team": "Mexico",        "away_team": "Sweden",        "odds_home": 2.20, "odds_draw": 3.10, "odds_away": 3.50},
        # Group G
        {"home_team": "Belgium",       "away_team": "Panama",        "odds_home": 1.17, "odds_draw": 7.50, "odds_away": 19.00},
        {"home_team": "Tunisia",       "away_team": "England",       "odds_home": 5.00, "odds_draw": 3.80, "odds_away": 1.67},
        {"home_team": "Belgium",       "away_team": "Tunisia",       "odds_home": 1.33, "odds_draw": 5.25, "odds_away": 11.00},
        {"home_team": "England",       "away_team": "Panama",        "odds_home": 1.24, "odds_draw": 6.50, "odds_away": 15.00},
        {"home_team": "England",       "away_team": "Belgium",       "odds_home": 3.10, "odds_draw": 3.20, "odds_away": 2.40},
        {"home_team": "Panama",        "away_team": "Tunisia",       "odds_home": 2.80, "odds_draw": 3.00, "odds_away": 2.75},
        # Group H
        {"home_team": "Colombia",      "away_team": "Japan",         "odds_home": 1.53, "odds_draw": 3.90, "odds_away": 7.50},
        {"home_team": "Poland",        "away_team": "Senegal",       "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 4.75},
        {"home_team": "Japan",         "away_team": "Senegal",       "odds_home": 3.20, "odds_draw": 3.10, "odds_away": 2.35},
        {"home_team": "Poland",        "away_team": "Colombia",      "odds_home": 2.55, "odds_draw": 3.10, "odds_away": 2.90},
        {"home_team": "Japan",         "away_team": "Poland",        "odds_home": 3.40, "odds_draw": 3.10, "odds_away": 2.25},
        {"home_team": "Senegal",       "away_team": "Colombia",      "odds_home": 3.30, "odds_draw": 3.20, "odds_away": 2.25},

        # ============ 決勝トーナメント ============
        # ラウンド16
        {"home_team": "France",        "away_team": "Argentina",     "odds_home": 1.90, "odds_draw": 3.50, "odds_away": 4.33},
        {"home_team": "Uruguay",       "away_team": "Portugal",      "odds_home": 2.30, "odds_draw": 3.30, "odds_away": 3.20},
        {"home_team": "Spain",         "away_team": "Russia",        "odds_home": 1.35, "odds_draw": 4.75, "odds_away": 9.50},
        {"home_team": "Croatia",       "away_team": "Denmark",       "odds_home": 1.90, "odds_draw": 3.25, "odds_away": 4.50},
        {"home_team": "Brazil",        "away_team": "Mexico",        "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00},
        {"home_team": "Belgium",       "away_team": "Japan",         "odds_home": 1.42, "odds_draw": 4.50, "odds_away": 9.00},
        {"home_team": "Sweden",        "away_team": "Switzerland",   "odds_home": 2.00, "odds_draw": 3.30, "odds_away": 4.00},
        {"home_team": "Colombia",      "away_team": "England",       "odds_home": 2.95, "odds_draw": 3.25, "odds_away": 2.50},
        # 準々決勝
        {"home_team": "Uruguay",       "away_team": "France",        "odds_home": 3.60, "odds_draw": 3.40, "odds_away": 2.10},
        {"home_team": "Brazil",        "away_team": "Belgium",       "odds_home": 1.67, "odds_draw": 3.60, "odds_away": 5.50},
        {"home_team": "Sweden",        "away_team": "England",       "odds_home": 3.20, "odds_draw": 3.10, "odds_away": 2.40},
        {"home_team": "Russia",        "away_team": "Croatia",       "odds_home": 3.00, "odds_draw": 3.10, "odds_away": 2.55},
        # 準決勝
        {"home_team": "France",        "away_team": "Belgium",       "odds_home": 1.95, "odds_draw": 3.40, "odds_away": 4.33},
        {"home_team": "Croatia",       "away_team": "England",       "odds_home": 2.40, "odds_draw": 3.10, "odds_away": 3.10},
        # 3位決定戦
        {"home_team": "Belgium",       "away_team": "England",       "odds_home": 2.10, "odds_draw": 3.20, "odds_away": 3.60},
        # 決勝
        {"home_team": "France",        "away_team": "Croatia",       "odds_home": 1.80, "odds_draw": 3.50, "odds_away": 4.75},
    ]

    df = pd.DataFrame(odds_data)
    print(f"Saving 2018 Russia World Cup odds ({len(df)} matches) to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
