#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_injuries_2026.py - API-Football から2026年W杯の負傷者・スタメン情報を取得する

事前準備:
  1. https://www.api-football.com/ で無料アカウント登録（無料枠: 100リクエスト/日）
  2. ダッシュボードのAPIキーを .env に追記:  API_FOOTBALL_KEY=xxxx

使い方:
  # 負傷者リストの取得 → data/raw/squad/injuries_2026.csv 更新 + ペナルティ推奨値の表示
  python src/pipeline/fetch_injuries_2026.py

  # 今日のW杯試合のスタメン確認（キックオフ約1時間前から取得可能）
  python src/pipeline/fetch_injuries_2026.py --lineups
"""
import os
import sys
import argparse
from collections import defaultdict
import pandas as pd
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
API_BASE = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1      # API-Football における FIFA World Cup のリーグID
WC_SEASON = 2026

# API-Football のチーム名 → 本プロジェクトの名称への名寄せ
TEAM_NAME_MAP = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Ireland": "Republic of Ireland",
}


def normalize(team):
    return TEAM_NAME_MAP.get(team, team)


def load_api_key():
    key = os.environ.get("API_FOOTBALL_KEY")
    if key:
        return key
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("API_FOOTBALL_KEY=") :
                val = line.split("=", 1)[1].strip().strip('"\'')
                if val:
                    return val
    return None


def api_get(key, endpoint, params):
    resp = requests.get(f"{API_BASE}/{endpoint}",
                        headers={"x-apisports-key": key},
                        params=params, timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR] APIリクエスト失敗 ({endpoint}): HTTP {resp.status_code}\n{resp.text[:300]}")
        sys.exit(1)
    data = resp.json()
    if data.get("errors"):
        print(f"[ERROR] APIエラー ({endpoint}): {data['errors']}")
        sys.exit(1)
    return data.get("response", [])


def fetch_injuries(key):
    """W杯エントリー選手の負傷情報を取得して保存・集計する"""
    out_path = os.path.join(BASE_DIR, "data/raw/squad/injuries_2026.csv")

    print(f"Fetching injuries (league={WC_LEAGUE_ID}, season={WC_SEASON})...")
    items = api_get(key, "injuries", {"league": WC_LEAGUE_ID, "season": WC_SEASON})
    print(f"  {len(items)} 件取得")

    rows = []
    for it in items:
        rows.append({
            "team": normalize(it.get("team", {}).get("name", "")),
            "player": it.get("player", {}).get("name", ""),
            "type": it.get("player", {}).get("type", ""),       # Missing Fixture / Questionable
            "reason": it.get("player", {}).get("reason", ""),
            "fixture_date": (it.get("fixture", {}).get("date") or "")[:10],
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player", "fixture_date"])
    df = df.sort_values(["team", "player"]) if len(df) else df
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if len(df) == 0:
        print("負傷者情報なし（大会序盤はデータ反映が遅れる場合があります）")
        return

    # チーム別の欠場見込み人数からペナルティの目安を提示
    print("\n=== チーム別 欠場見込み（type=Missing Fixture のみ） ===")
    missing = df[df["type"].str.contains("Missing", case=False, na=False)]
    counts = missing.groupby("team")["player"].nunique().sort_values(ascending=False)
    pen_path = "data/raw/squad/squad_penalties_2026.csv"
    print(f"{'チーム':<24}{'欠場数':>6}  ペナルティ目安(主力なら下げる)")
    for team, n in counts.items():
        suggested = max(0.85, 1.0 - 0.03 * n)
        players = ", ".join(missing[missing['team'] == team]['player'].unique()[:5])
        print(f"{team:<24}{n:>6}  {suggested:.2f}  ({players})")
    print(f"\n※ 主力（スタメン級）の離脱は1人あたり-5〜10%が目安。判断のうえ {pen_path} に記入し、")
    print("   predict_scores_2026.py / simulator_2026.py を再実行してください。")


def fetch_lineups(key):
    """今日のW杯試合のスタメンを表示する（キックオフ約1時間前から取得可能）"""
    from datetime import date
    today = date.today().isoformat()
    print(f"Fetching today's fixtures ({today})...")
    fixtures = api_get(key, "fixtures",
                       {"league": WC_LEAGUE_ID, "season": WC_SEASON, "date": today})
    if not fixtures:
        print("本日のW杯試合はありません。")
        return

    for fx in fixtures:
        fid = fx["fixture"]["id"]
        home = normalize(fx["teams"]["home"]["name"])
        away = normalize(fx["teams"]["away"]["name"])
        kickoff = fx["fixture"]["date"]
        print(f"\n=== {home} vs {away} ({kickoff}) ===")

        lineups = api_get(key, "fixtures/lineups", {"fixture": fid})
        if not lineups:
            print("  スタメン未発表（キックオフ約1時間前に再実行してください）")
            continue
        for lu in lineups:
            team = normalize(lu["team"]["name"])
            coach = (lu.get("coach") or {}).get("name", "?")
            formation = lu.get("formation", "?")
            xi = [p["player"]["name"] for p in lu.get("startXI", [])]
            print(f"  {team} ({formation}, 監督: {coach})")
            print(f"    {', '.join(xi)}")


def main():
    parser = argparse.ArgumentParser(description="API-Football 負傷者・スタメン取得")
    parser.add_argument("--lineups", action="store_true", help="今日の試合のスタメンを表示")
    args = parser.parse_args()

    key = load_api_key()
    if not key:
        print("[ERROR] APIキーが設定されていません。以下の手順で設定してください:")
        print("  1. https://www.api-football.com/ で無料登録（無料枠100リクエスト/日）")
        print("  2. ダッシュボードに表示されるAPIキーを .env に追記:")
        print("       API_FOOTBALL_KEY=あなたのキー")
        sys.exit(1)

    if args.lineups:
        fetch_lineups(key)
    else:
        fetch_injuries(key)


if __name__ == "__main__":
    main()
