#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_winner_bias.py - WINNER 18択市場のバイアス分析

winner_matches/*_ev.csv（モデル確率と購入時オッズ）を集計し、
選択肢カテゴリごとに「市場インプライド確率 vs モデル確率」を比較して
WINNER市場がどこを過大/過小評価しているかを定量化する。
試合結果が判明しているもの（buy_status.csv の match_result または results.csv）があれば
実績的中率も併記する。

使い方:
  python src/predict/analyze_winner_bias.py
"""
import os
import sys
import glob
import re
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator_2026 import GROUPS_2026  # noqa: E402

# クリーン名（小文字・_区切り） → 正式名
CLEAN_TO_TEAM = {t.lower().replace(' ', '_'): t
                 for ts in GROUPS_2026.values() for t in ts}


def parse_teams_from_filename(base):
    """'04_winner_united_states_paraguay_ev.csv' → ('United States', 'Paraguay')"""
    m = re.match(r"\d+_winner_(.+)_ev\.csv", base)
    if not m:
        return None, None
    middle = m.group(1)
    parts = middle.split('_')
    # 既知チーム名に一致する分割点を探す
    for i in range(1, len(parts)):
        h, a = '_'.join(parts[:i]), '_'.join(parts[i:])
        if h in CLEAN_TO_TEAM and a in CLEAN_TO_TEAM:
            return CLEAN_TO_TEAM[h], CLEAN_TO_TEAM[a]
    return None, None

# 選択肢名 → カテゴリ分類
def categorize(selection):
    s = str(selection)
    if 'その他' in s:
        if '引き分け' in s:
            return '引き分け その他(3+)'
        return '勝利 その他(4+)'
    if '引き分け' in s:
        return '引き分け 通常(0-0/1-1/2-2)'
    m = re.search(r'(\d+)\s*-\s*(\d+)', s)
    if m:
        total = int(m.group(1)) + int(m.group(2))
        return '勝利 ロースコア(計1-2点)' if total <= 2 else '勝利 ハイスコア(計3点以上)'
    return 'その他'


def parse_score(text):
    m = re.search(r'(\d+)\s*-\s*(\d+)', str(text))
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_results_lookup():
    """確定済みの試合スコアを (home, away) -> (h, a) で返す（results.csv ベース）"""
    res = pd.read_csv(os.path.join(BASE_DIR, "data/raw/match/results.csv"))
    res = res[(res['tournament'] == 'FIFA World Cup') & (res['date'] >= '2026-06-01')]
    res = res.dropna(subset=['home_score', 'away_score'])
    lookup = {}
    for r in res.itertuples():
        lookup[(r.home_team, r.away_team)] = (int(r.home_score), int(r.away_score))
    # buy_status の手入力結果も補完
    buy_path = os.path.join(BASE_DIR, "data/buy_status.csv")
    if os.path.exists(buy_path):
        buy = pd.read_csv(buy_path)
        if 'match_result' in buy.columns:
            for r in buy.itertuples():
                sc = parse_score(getattr(r, 'match_result', ''))
                if sc and (r.home_team, r.away_team) not in lookup:
                    lookup[(r.home_team, r.away_team)] = sc
    return lookup


def selection_hit(selection, home, away, score):
    """選択肢が的中したか（settle_bets と同じ判定ロジックの簡易版）"""
    h, a = score
    target = parse_score(selection)
    if target:
        return int((h, a) == target)
    s = str(selection)
    if '引き分け' in s and 'その他' in s:
        return int(h == a and h >= 3)
    if 'その他' in s:
        # 「<チーム名> その他(4得点以上)」
        if s.startswith(str(home)):
            return int(h > a and h >= 4)
        if s.startswith(str(away)):
            return int(a > h and a >= 4)
    return 0


def main():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "data/processed/2026/winner_matches/*_ev.csv")))
    if not files:
        print("winner_matches/ にEVファイルがありません。")
        return

    results = load_results_lookup()
    rows = []
    for path in files:
        base = os.path.basename(path)
        df = pd.read_csv(path)
        home, away = parse_teams_from_filename(base)
        if home is None:
            print(f"  [WARNING] チーム名を復元できずスキップ: {base}")
            continue
        score = results.get((home, away))

        for r in df.itertuples():
            odds = float(r.Odds)
            if odds <= 1.0:
                continue  # オッズ未入力
            rows.append({
                'match': f"{home} vs {away}",
                'selection': r.Selection,
                'category': categorize(r.Selection),
                'odds': odds,
                'implied_prob': 1.0 / odds,
                'model_prob': getattr(r, '_3') / 100.0,  # Probability(%)
                'ev': getattr(r, '_4'),                  # Expected Value (EV)
                'hit': selection_hit(r.Selection, home, away, score) if score else None,
                'settled': score is not None,
            })

    d = pd.DataFrame(rows)
    # 各試合の還元率（インプライド確率合計の逆数）
    overround = d.groupby('match')['implied_prob'].sum()
    print("=== 試合別の市場マージン（インプライド確率合計。1.0超過分が控除率相当） ===")
    for match, s in overround.items():
        print(f"  {match:<45} {s:.2f} (還元率 {100/s:.0f}%)")

    print(f"\n=== カテゴリ別バイアス（{d['match'].nunique()}試合, {len(d)}選択肢） ===")
    g = d.groupby('category').agg(
        n=('selection', 'size'),
        市場確率合計=('implied_prob', 'sum'),
        モデル確率合計=('model_prob', 'sum'),
        平均EV=('ev', 'mean'),
    )
    g['市場/モデル'] = g['市場確率合計'] / g['モデル確率合計']
    print(g.round(3).to_string())
    print("\n※ 市場/モデル > 1 = 市場が過大評価（買うと不利）/ < 1 = 市場が過小評価（妙味）")

    settled = d[d['settled']]
    if len(settled):
        print(f"\n=== 実績（結果確定 {settled['match'].nunique()}試合分） ===")
        gs = settled.groupby('category').agg(
            n=('selection', 'size'),
            モデル確率計=('model_prob', 'sum'),
            的中数=('hit', 'sum'),
        )
        print(gs.round(2).to_string())

    out = os.path.join(BASE_DIR, "data/processed/2026/winner_bias_analysis.csv")
    d.to_csv(out, index=False)
    print(f"\n明細を保存: {out}")


if __name__ == "__main__":
    main()
