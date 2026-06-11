"""
prepare_odds_2026_groups.py
2026年W杯 グループステージ全72試合のブックメーカーオッズを定義する。
実際のオッズが取得できたものはreal、それ以外は
モデル確率に典型的なオーバーラウンド（6%）を適用したestimatedを設定。

オッズを実際のブックメーカー値に更新したい場合はこのファイルを直接編集してください。
"""
import os
import pandas as pd

# ============================================================
# 各グループの試合オッズ定義
# ============================================================
# odds_home, odds_draw, odds_away はブックメーカーの倍率 (欧州式小数オッズ)
# source: 'real' = 実際のブックメーカー値, 'estimated' = モデル推定値

ODDS_DATA = [
    # ==== Group A: Mexico, South Africa, South Korea, Czech Republic ====
    # 6/11 開幕戦
    {"group": "A", "home_team": "Mexico",       "away_team": "South Africa", "odds_home": 1.75, "odds_draw": 3.30, "odds_away": 5.00, "source": "estimated"},
    {"group": "A", "home_team": "South Korea",  "away_team": "Czech Republic","odds_home": 2.75, "odds_draw": 3.25, "odds_away": 2.88, "source": "real"},
    {"group": "A", "home_team": "Mexico",       "away_team": "Czech Republic","odds_home": 1.85, "odds_draw": 3.40, "odds_away": 4.75, "source": "estimated"},
    {"group": "A", "home_team": "South Africa", "away_team": "South Korea",  "odds_home": 3.80, "odds_draw": 3.20, "odds_away": 2.10, "source": "estimated"},
    {"group": "A", "home_team": "Mexico",       "away_team": "South Korea",  "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 4.80, "source": "estimated"},
    {"group": "A", "home_team": "Czech Republic","away_team": "South Africa","odds_home": 2.00, "odds_draw": 3.30, "odds_away": 4.00, "source": "estimated"},

    # ==== Group B: Canada, Bosnia and Herzegovina, Qatar, Switzerland ====
    {"group": "B", "home_team": "Canada",       "away_team": "Bosnia and Herzegovina","odds_home": 2.10, "odds_draw": 3.20, "odds_away": 3.60, "source": "estimated"},
    {"group": "B", "home_team": "Switzerland",  "away_team": "Qatar",        "odds_home": 1.60, "odds_draw": 3.70, "odds_away": 6.50, "source": "estimated"},
    {"group": "B", "home_team": "Canada",       "away_team": "Qatar",        "odds_home": 1.55, "odds_draw": 3.80, "odds_away": 7.00, "source": "estimated"},
    {"group": "B", "home_team": "Bosnia and Herzegovina","away_team": "Switzerland","odds_home": 3.80, "odds_draw": 3.25, "odds_away": 2.10, "source": "estimated"},
    {"group": "B", "home_team": "Canada",       "away_team": "Switzerland",  "odds_home": 2.50, "odds_draw": 3.10, "odds_away": 3.00, "source": "estimated"},
    {"group": "B", "home_team": "Qatar",        "away_team": "Bosnia and Herzegovina","odds_home": 3.20, "odds_draw": 3.20, "odds_away": 2.40, "source": "estimated"},

    # ==== Group C: Brazil, Morocco, Haiti, Scotland ====
    {"group": "C", "home_team": "Brazil",       "away_team": "Morocco",      "odds_home": 1.85, "odds_draw": 3.40, "odds_away": 4.75, "source": "estimated"},
    {"group": "C", "home_team": "Haiti",        "away_team": "Scotland",     "odds_home": 3.80, "odds_draw": 3.25, "odds_away": 2.10, "source": "estimated"},
    {"group": "C", "home_team": "Brazil",       "away_team": "Scotland",     "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00, "source": "estimated"},
    {"group": "C", "home_team": "Morocco",      "away_team": "Haiti",        "odds_home": 1.57, "odds_draw": 3.70, "odds_away": 6.50, "source": "estimated"},
    {"group": "C", "home_team": "Brazil",       "away_team": "Haiti",        "odds_home": 1.20, "odds_draw": 6.00, "odds_away": 15.00,"source": "estimated"},
    {"group": "C", "home_team": "Scotland",     "away_team": "Morocco",      "odds_home": 3.60, "odds_draw": 3.20, "odds_away": 2.20, "source": "estimated"},

    # ==== Group D: United States, Paraguay, Australia, Turkey ====
    {"group": "D", "home_team": "United States","away_team": "Paraguay",     "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 5.00, "source": "estimated"},
    {"group": "D", "home_team": "Australia",    "away_team": "Turkey",       "odds_home": 2.80, "odds_draw": 3.10, "odds_away": 2.80, "source": "estimated"},
    {"group": "D", "home_team": "United States","away_team": "Australia",    "odds_home": 2.10, "odds_draw": 3.20, "odds_away": 3.70, "source": "estimated"},
    {"group": "D", "home_team": "Paraguay",     "away_team": "Turkey",       "odds_home": 2.50, "odds_draw": 3.10, "odds_away": 3.00, "source": "estimated"},
    {"group": "D", "home_team": "United States","away_team": "Turkey",       "odds_home": 1.90, "odds_draw": 3.35, "odds_away": 4.50, "source": "estimated"},
    {"group": "D", "home_team": "Paraguay",     "away_team": "Australia",    "odds_home": 2.60, "odds_draw": 3.10, "odds_away": 2.90, "source": "estimated"},

    # ==== Group E: Germany, Curaçao, Ivory Coast, Ecuador ====
    {"group": "E", "home_team": "Germany",      "away_team": "Curaçao",      "odds_home": 1.15, "odds_draw": 7.50, "odds_away": 20.00,"source": "estimated"},
    {"group": "E", "home_team": "Ivory Coast",  "away_team": "Ecuador",      "odds_home": 2.50, "odds_draw": 3.10, "odds_away": 3.00, "source": "estimated"},
    {"group": "E", "home_team": "Germany",      "away_team": "Ecuador",      "odds_home": 1.50, "odds_draw": 4.00, "odds_away": 7.50, "source": "estimated"},
    {"group": "E", "home_team": "Curaçao",      "away_team": "Ivory Coast",  "odds_home": 4.50, "odds_draw": 3.40, "odds_away": 1.90, "source": "estimated"},
    {"group": "E", "home_team": "Germany",      "away_team": "Ivory Coast",  "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00, "source": "estimated"},
    {"group": "E", "home_team": "Ecuador",      "away_team": "Curaçao",      "odds_home": 1.55, "odds_draw": 3.80, "odds_away": 7.00, "source": "estimated"},

    # ==== Group F: Netherlands, Japan, Tunisia, Sweden ====
    {"group": "F", "home_team": "Netherlands",  "away_team": "Japan",        "odds_home": 1.77, "odds_draw": 3.50, "odds_away": 3.60, "source": "real"},
    {"group": "F", "home_team": "Sweden",       "away_team": "Tunisia",      "odds_home": 2.00, "odds_draw": 3.30, "odds_away": 4.00, "source": "estimated"},
    {"group": "F", "home_team": "Netherlands",  "away_team": "Sweden",       "odds_home": 1.65, "odds_draw": 3.70, "odds_away": 5.50, "source": "estimated"},
    {"group": "F", "home_team": "Tunisia",      "away_team": "Japan",        "odds_home": 4.50, "odds_draw": 3.40, "odds_away": 1.80, "source": "estimated"},
    {"group": "F", "home_team": "Japan",        "away_team": "Sweden",       "odds_home": 2.50, "odds_draw": 3.10, "odds_away": 3.00, "source": "estimated"},
    {"group": "F", "home_team": "Tunisia",      "away_team": "Netherlands",  "odds_home": 8.00, "odds_draw": 4.50, "odds_away": 1.40, "source": "estimated"},

    # ==== Group G: Belgium, Egypt, Iran, New Zealand ====
    {"group": "G", "home_team": "Belgium",      "away_team": "Egypt",        "odds_home": 1.60, "odds_draw": 3.75, "odds_away": 6.50, "source": "estimated"},
    {"group": "G", "home_team": "Iran",         "away_team": "New Zealand",  "odds_home": 1.65, "odds_draw": 3.60, "odds_away": 6.00, "source": "estimated"},
    {"group": "G", "home_team": "Belgium",      "away_team": "Iran",         "odds_home": 1.70, "odds_draw": 3.50, "odds_away": 5.50, "source": "estimated"},
    {"group": "G", "home_team": "Egypt",        "away_team": "New Zealand",  "odds_home": 1.85, "odds_draw": 3.40, "odds_away": 4.75, "source": "estimated"},
    {"group": "G", "home_team": "Belgium",      "away_team": "New Zealand",  "odds_home": 1.30, "odds_draw": 5.25, "odds_away": 11.00,"source": "estimated"},
    {"group": "G", "home_team": "Iran",         "away_team": "Egypt",        "odds_home": 2.30, "odds_draw": 3.20, "odds_away": 3.25, "source": "estimated"},

    # ==== Group H: Spain, Cape Verde, Saudi Arabia, Uruguay ====
    {"group": "H", "home_team": "Spain",        "away_team": "Cape Verde",   "odds_home": 1.12, "odds_draw": 8.00, "odds_away": 22.00,"source": "estimated"},
    {"group": "H", "home_team": "Saudi Arabia", "away_team": "Uruguay",      "odds_home": 2.80, "odds_draw": 3.10, "odds_away": 2.80, "source": "estimated"},
    {"group": "H", "home_team": "Spain",        "away_team": "Uruguay",      "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00, "source": "estimated"},
    {"group": "H", "home_team": "Cape Verde",   "away_team": "Saudi Arabia", "odds_home": 3.80, "odds_draw": 3.25, "odds_away": 2.10, "source": "estimated"},
    {"group": "H", "home_team": "Spain",        "away_team": "Saudi Arabia", "odds_home": 1.33, "odds_draw": 5.00, "odds_away": 11.00,"source": "estimated"},
    {"group": "H", "home_team": "Uruguay",      "away_team": "Cape Verde",   "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00, "source": "estimated"},

    # ==== Group I: France, Senegal, Norway, Iraq ====
    {"group": "I", "home_team": "France",       "away_team": "Senegal",      "odds_home": 1.57, "odds_draw": 3.80, "odds_away": 6.50, "source": "estimated"},
    {"group": "I", "home_team": "Norway",       "away_team": "Iraq",         "odds_home": 1.55, "odds_draw": 3.80, "odds_away": 7.00, "source": "estimated"},
    {"group": "I", "home_team": "France",       "away_team": "Norway",       "odds_home": 1.67, "odds_draw": 3.60, "odds_away": 5.50, "source": "estimated"},
    {"group": "I", "home_team": "Senegal",      "away_team": "Iraq",         "odds_home": 1.67, "odds_draw": 3.60, "odds_away": 5.50, "source": "estimated"},
    {"group": "I", "home_team": "France",       "away_team": "Iraq",         "odds_home": 1.25, "odds_draw": 6.00, "odds_away": 13.00,"source": "estimated"},
    {"group": "I", "home_team": "Norway",       "away_team": "Senegal",      "odds_home": 2.20, "odds_draw": 3.20, "odds_away": 3.50, "source": "estimated"},

    # ==== Group J: Argentina, Algeria, Austria, Jordan ====
    {"group": "J", "home_team": "Argentina",    "away_team": "Algeria",      "odds_home": 1.30, "odds_draw": 5.25, "odds_away": 12.00,"source": "estimated"},
    {"group": "J", "home_team": "Austria",      "away_team": "Jordan",       "odds_home": 1.70, "odds_draw": 3.50, "odds_away": 5.50, "source": "estimated"},
    {"group": "J", "home_team": "Argentina",    "away_team": "Austria",      "odds_home": 1.50, "odds_draw": 4.00, "odds_away": 7.50, "source": "estimated"},
    {"group": "J", "home_team": "Algeria",      "away_team": "Jordan",       "odds_home": 1.90, "odds_draw": 3.35, "odds_away": 4.50, "source": "estimated"},
    {"group": "J", "home_team": "Argentina",    "away_team": "Jordan",       "odds_home": 1.20, "odds_draw": 6.50, "odds_away": 16.00,"source": "estimated"},
    {"group": "J", "home_team": "Austria",      "away_team": "Algeria",      "odds_home": 2.20, "odds_draw": 3.20, "odds_away": 3.50, "source": "estimated"},

    # ==== Group K: Portugal, Uzbekistan, Colombia, DR Congo ====
    {"group": "K", "home_team": "Portugal",     "away_team": "Uzbekistan",   "odds_home": 1.40, "odds_draw": 4.50, "odds_away": 9.00, "source": "estimated"},
    {"group": "K", "home_team": "Colombia",     "away_team": "DR Congo",     "odds_home": 1.53, "odds_draw": 3.80, "odds_away": 7.00, "source": "estimated"},
    {"group": "K", "home_team": "Portugal",     "away_team": "Colombia",     "odds_home": 2.10, "odds_draw": 3.20, "odds_away": 3.70, "source": "estimated"},
    {"group": "K", "home_team": "Uzbekistan",   "away_team": "DR Congo",     "odds_home": 1.90, "odds_draw": 3.35, "odds_away": 4.50, "source": "estimated"},
    {"group": "K", "home_team": "Portugal",     "away_team": "DR Congo",     "odds_home": 1.22, "odds_draw": 6.00, "odds_away": 14.00,"source": "estimated"},
    {"group": "K", "home_team": "Colombia",     "away_team": "Uzbekistan",   "odds_home": 1.67, "odds_draw": 3.60, "odds_away": 5.50, "source": "estimated"},

    # ==== Group L: England, Croatia, Ghana, Panama ====
    {"group": "L", "home_team": "England",      "away_team": "Croatia",      "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 5.00, "source": "estimated"},
    {"group": "L", "home_team": "Ghana",        "away_team": "Panama",       "odds_home": 1.90, "odds_draw": 3.35, "odds_away": 4.50, "source": "estimated"},
    {"group": "L", "home_team": "England",      "away_team": "Ghana",        "odds_home": 1.45, "odds_draw": 4.25, "odds_away": 8.00, "source": "estimated"},
    {"group": "L", "home_team": "Croatia",      "away_team": "Panama",       "odds_home": 1.57, "odds_draw": 3.80, "odds_away": 6.50, "source": "estimated"},
    {"group": "L", "home_team": "England",      "away_team": "Panama",       "odds_home": 1.22, "odds_draw": 6.00, "odds_away": 14.00,"source": "estimated"},
    {"group": "L", "home_team": "Croatia",      "away_team": "Ghana",        "odds_home": 1.80, "odds_draw": 3.40, "odds_away": 5.00, "source": "estimated"},
]


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_path = os.path.join(base_dir, "data/raw/odds/odds_groups_2026.csv")

    df = pd.DataFrame(ODDS_DATA)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} match odds to {output_path}")
    real_count = len(df[df['source'] == 'real'])
    print(f"  Real odds: {real_count} / Estimated: {len(df) - real_count}")


if __name__ == "__main__":
    main()
