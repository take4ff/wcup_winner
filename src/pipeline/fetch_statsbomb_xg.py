#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_statsbomb_xg.py - StatsBombオープンデータから過去W杯のチーム別xGを抽出する

StatsBomb open data (https://github.com/statsbomb/open-data) には
2018年・2022年W杯の全試合イベントデータ（xG付きシュート）が無料公開されている。
本スクリプトは試合ごとの両チーム合計xGを抽出して
data/raw/xg/statsbomb_wc_xg.csv に保存する（xG導入の効果検証用）。

使い方:
  python src/pipeline/fetch_statsbomb_xg.py
"""
import os
import time
import pandas as pd
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMP_ID = 43  # FIFA World Cup
SEASONS = {3: 2018, 106: 2022}  # season_id -> 大会年

# StatsBomb のチーム名 → 本プロジェクトの名称
TEAM_NAME_MAP = {
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "United States": "United States",
    "IR Iran": "Iran",
}


def normalize(team):
    return TEAM_NAME_MAP.get(team, team)


def fetch_json(url, retries=3):
    for i in range(retries):
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"取得失敗: {url} (HTTP {resp.status_code})")


def main():
    out_dir = os.path.join(BASE_DIR, "data/raw/xg")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "statsbomb_wc_xg.csv")

    records = []
    for season_id, year in SEASONS.items():
        print(f"=== {year} World Cup (season_id={season_id}) ===")
        matches = fetch_json(f"{RAW_BASE}/matches/{COMP_ID}/{season_id}.json")
        print(f"  {len(matches)} matches")

        for k, m in enumerate(matches, 1):
            mid = m["match_id"]
            home = normalize(m["home_team"]["home_team_name"])
            away = normalize(m["away_team"]["away_team_name"])
            date = m["match_date"]

            events = fetch_json(f"{RAW_BASE}/events/{mid}.json")
            xg = {home: 0.0, away: 0.0}
            shots = {home: 0, away: 0}
            for ev in events:
                if ev.get("type", {}).get("name") == "Shot":
                    team = normalize(ev["team"]["name"])
                    val = ev.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
                    if team in xg:
                        xg[team] += val
                        shots[team] += 1

            records.append({
                "year": year, "date": date,
                "home_team": home, "away_team": away,
                "home_score": m["home_score"], "away_score": m["away_score"],
                "home_xg": round(xg[home], 3), "away_xg": round(xg[away], 3),
                "home_shots": shots[home], "away_shots": shots[away],
            })
            if k % 16 == 0:
                print(f"  ... {k}/{len(matches)} done")

    df = pd.DataFrame(records).sort_values(["year", "date"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} matches to {out_path}")


if __name__ == "__main__":
    main()
