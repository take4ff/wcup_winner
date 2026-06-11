#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_odds_2026.py
The-Odds-API (https://the-odds-api.com/) を利用して、
2026年W杯グループステージの実際のブックメーカーオッズを取得し、
data/raw/odds/odds_groups_2026.csv を更新する。

使い方:
  # 環境変数または.envにキーを設定している場合
  python src/odds/fetch_odds_2026.py
  
  # 直接APIキーを指定する場合
  python src/odds/fetch_odds_2026.py --api_key YOUR_KEY
"""
import os
import sys
import argparse
import pandas as pd
import requests

# 国名・チーム名の表記ゆれ名寄せマッピング
TEAM_NAME_MAP = {
    "USA": "United States",
    "United States": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Cape Verde": "Cape Verde",
}


def load_api_key(api_key_arg=None):
    """APIキーを引数、環境変数、または.envファイルから読み込む"""
    if api_key_arg:
        return api_key_arg

    # 環境変数から取得
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key

    # .env ファイルから読み込み試行
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        print(f"Loading API key from {env_path}...")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "THE_ODDS_API_KEY":
                        val = v.strip()
                        if val.startswith(('"', "'")) and val.endswith(('"', "'")):
                            val = val[1:-1]
                        return val
    return None


def extract_h2h_odds(bm, home_name, away_name):
    """ブックメーカーのデータからH2H（勝敗分け）オッズを抽出する"""
    markets = bm.get("markets", [])
    if not markets:
        return None, None, None

    h2h_market = None
    for mk in markets:
        if mk.get("key") == "h2h":
            h2h_market = mk
            break

    if not h2h_market:
        return None, None, None

    outcomes = h2h_market.get("outcomes", [])
    odds_h, odds_d, odds_a = None, None, None
    for ot in outcomes:
        name = ot.get("name")
        price = ot.get("price")
        if name == home_name:
            odds_h = price
        elif name == away_name:
            odds_a = price
        elif name in ["Draw", "draw", "Tie", "tie"]:
            odds_d = price

    if odds_h is not None and odds_a is not None and odds_d is not None:
        return odds_h, odds_d, odds_a
    return None, None, None


def make_key(h, a):
    """対称な比較のためのマージキー作成"""
    return f"{min(h, a)}_{max(h, a)}"


def main():
    parser = argparse.ArgumentParser(description="The-Odds-API 2026 W杯オッズ取得スクリプト")
    parser.add_argument("--api_key", type=str, help="The-Odds-APIのAPIキー")
    parser.add_argument("--regions", type=str, default="eu", help="取得対象のブックメーカー地域 (eu, us, uk, auなど)")
    parser.add_argument("--bookmaker", type=str, default="pinnacle", help="最優先で使用するブックメーカーキー (pinnacle, bet365など)")
    args = parser.parse_args()

    # APIキーの読み込み
    api_key = load_api_key(args.api_key)
    if not api_key or api_key == "your_api_key_here":
        print("[ERROR] APIキーが設定されていません。")
        print("以下いずれかの方法で設定してください:")
        print("  1. プロジェクトのルートに .env ファイルを作成し、THE_ODDS_API_KEY=xxx と記述する")
        print("  2. 環境変数 THE_ODDS_API_KEY を設定する")
        print("  3. 実行引数に --api_key YOUR_KEY を渡す")
        sys.exit(1)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_path = os.path.join(base_dir, "data/raw/odds/odds_groups_2026.csv")

    if not os.path.exists(output_path):
        print(f"[ERROR] テンプレートとなるオッズファイルが存在しません: {output_path}")
        sys.exit(1)

    # APIリクエスト
    url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"
    params = {
        "apiKey": api_key,
        "regions": args.regions,
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    print(f"Fetching live odds from: {url}")
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"[ERROR] APIリクエストが失敗しました。ステータスコード: {response.status_code}")
            print(f"詳細: {response.text}")
            sys.exit(1)
        data = response.json()
    except Exception as e:
        print(f"[ERROR] 通信エラーが発生しました: {e}")
        sys.exit(1)

    print(f"Successfully retrieved {len(data)} matches from API.")

    # APIデータのパースと名寄せ
    extracted_odds = []
    for match in data:
        home = match.get("home_team")
        away = match.get("away_team")

        # 名寄せ適用
        home_norm = TEAM_NAME_MAP.get(home, home)
        away_norm = TEAM_NAME_MAP.get(away, away)

        bookmakers = match.get("bookmakers", [])
        if not bookmakers:
            continue

        odds_h, odds_d, odds_a = None, None, None
        selected_bm = None

        # 1. 指定された優先ブックメーカー（例: pinnacle）を探す
        for bm in bookmakers:
            if bm.get("key") == args.bookmaker:
                selected_bm = args.bookmaker
                odds_h, odds_d, odds_a = extract_h2h_odds(bm, home, away)
                break

        # 2. なければ Bet365 を探す
        if odds_h is None and args.bookmaker != "bet365":
            for bm in bookmakers:
                if bm.get("key") == "bet365":
                    selected_bm = "bet365"
                    odds_h, odds_d, odds_a = extract_h2h_odds(bm, home, away)
                    break

        # 3. それでもなければ、利用可能な全ブックメーカーの平均をとる
        if odds_h is None:
            bm_odds_list = []
            for bm in bookmakers:
                oh, od, oa = extract_h2h_odds(bm, home, away)
                if oh is not None:
                    bm_odds_list.append((oh, od, oa))
            if bm_odds_list:
                selected_bm = "average"
                odds_h = sum(x[0] for x in bm_odds_list) / len(bm_odds_list)
                odds_d = sum(x[1] for x in bm_odds_list) / len(bm_odds_list)
                odds_a = sum(x[2] for x in bm_odds_list) / len(bm_odds_list)

        if odds_h is not None:
            extracted_odds.append({
                "home_team": home_norm,
                "away_team": away_norm,
                "odds_home": round(odds_h, 3),
                "odds_draw": round(odds_d, 3),
                "odds_away": round(odds_a, 3),
                "bm_source": selected_bm
            })

    print(f"Parsed {len(extracted_odds)} match odds after name normalization.")

    # CSVの更新
    df_existing = pd.read_csv(output_path)
    df_existing['key'] = df_existing.apply(lambda r: make_key(r['home_team'], r['away_team']), axis=1)

    api_dict = {}
    for item in extracted_odds:
        k = make_key(item['home_team'], item['away_team'])
        api_dict[k] = item

    updated_count = 0
    for idx, row in df_existing.iterrows():
        k = row['key']
        if k in api_dict:
            api_item = api_dict[k]

            # ホームとアウェイが逆の場合オッズを入れ替える
            if row['home_team'] == api_item['home_team']:
                df_existing.at[idx, 'odds_home'] = api_item['odds_home']
                df_existing.at[idx, 'odds_away'] = api_item['odds_away']
            else:
                df_existing.at[idx, 'odds_home'] = api_item['odds_away']
                df_existing.at[idx, 'odds_away'] = api_item['odds_home']

            df_existing.at[idx, 'odds_draw'] = api_item['odds_draw']
            df_existing.at[idx, 'source'] = 'real'
            updated_count += 1
        else:
            # マッチしない場合、かつ以前 'real' だったものは estimated に戻さない（そのままにする）
            pass

    # キー列を削除して保存
    df_existing = df_existing.drop(columns=['key'])
    df_existing.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] {output_path} の更新が完了しました。")
    print(f"  更新された試合数: {updated_count} / {len(df_existing)}")
    real_total = len(df_existing[df_existing['source'] == 'real'])
    print(f"  現在の実オッズ総数: {real_total} / {len(df_existing)}")


if __name__ == "__main__":
    main()
