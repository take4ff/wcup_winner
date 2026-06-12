#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_xg_value.py - xG導入の効果検証（第1段階）

問い: 「それまでの試合のxG平均」は「それまでの試合の得点平均」よりも、
      次の試合の得点をよく予測するか？

StatsBombの2018・2022年W杯データ（statsbomb_wc_xg.csv）を使い、
大会内ウォークフォワード（各チームのt戦目を1..t-1戦目の統計で予測）で
ピアソン相関と平均絶対誤差を比較する。

使い方:
  python src/pipeline/fetch_statsbomb_xg.py   # 先にデータ取得
  python src/backtest/analyze_xg_value.py
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def main():
    path = os.path.join(BASE_DIR, "data/raw/xg/statsbomb_wc_xg.csv")
    if not os.path.exists(path):
        print("先に fetch_statsbomb_xg.py を実行してください。")
        return
    df = pd.read_csv(path)

    # チーム視点に展開
    rows = []
    for r in df.itertuples():
        rows.append({'year': r.year, 'date': r.date, 'team': r.home_team,
                     'goals': r.home_score, 'xg': r.home_xg,
                     'conceded': r.away_score, 'xga': r.away_xg})
        rows.append({'year': r.year, 'date': r.date, 'team': r.away_team,
                     'goals': r.away_score, 'xg': r.away_xg,
                     'conceded': r.home_score, 'xga': r.home_xg})
    t = pd.DataFrame(rows).sort_values(['year', 'team', 'date'])

    # 大会内ウォークフォワード: t戦目の得点を 1..t-1戦目の平均で予測
    g = t.groupby(['year', 'team'])
    t['pred_goals_from_goals'] = g['goals'].transform(lambda s: s.shift(1).expanding().mean())
    t['pred_goals_from_xg'] = g['xg'].transform(lambda s: s.shift(1).expanding().mean())
    t['pred_conc_from_conc'] = g['conceded'].transform(lambda s: s.shift(1).expanding().mean())
    t['pred_conc_from_xga'] = g['xga'].transform(lambda s: s.shift(1).expanding().mean())

    ev = t.dropna(subset=['pred_goals_from_goals', 'pred_goals_from_xg'])
    print(f"評価対象: {len(ev)} チーム試合 (2大会, 各チーム2戦目以降)\n")

    def report(label, pred_col, target_col):
        corr = ev[pred_col].corr(ev[target_col])
        mae = (ev[pred_col] - ev[target_col]).abs().mean()
        print(f"  {label:<28} 相関 {corr:+.3f} | MAE {mae:.3f}")

    print("=== 次戦の得点の予測力 ===")
    report("過去の得点平均 → 得点", 'pred_goals_from_goals', 'goals')
    report("過去のxG平均   → 得点", 'pred_goals_from_xg', 'goals')

    print("\n=== 次戦の失点の予測力 ===")
    report("過去の失点平均 → 失点", 'pred_conc_from_conc', 'conceded')
    report("過去の被xG平均 → 失点", 'pred_conc_from_xga', 'conceded')

    # 同一試合内の「内容と結果の乖離」: xGと得点の相関
    print("\n=== 参考: 同一試合内の相関 ===")
    print(f"  xG vs 実得点: {t['xg'].corr(t['goals']):+.3f}")

    # 大会間の持ち越し: 2018年の大会平均xG → 2022年の大会平均得点（出場が重なるチーム）
    agg = t.groupby(['year', 'team']).agg(mean_goals=('goals', 'mean'),
                                          mean_xg=('xg', 'mean')).reset_index()
    m18 = agg[agg['year'] == 2018].set_index('team')
    m22 = agg[agg['year'] == 2022].set_index('team')
    both = m18.join(m22, lsuffix='_18', rsuffix='_22', how='inner')
    if len(both) >= 8:
        print(f"\n=== 大会間の持ち越し (2018→2022, n={len(both)}チーム) ===")
        print(f"  2018年の平均得点 → 2022年の平均得点: 相関 {both['mean_goals_18'].corr(both['mean_goals_22']):+.3f}")
        print(f"  2018年の平均xG   → 2022年の平均得点: 相関 {both['mean_xg_18'].corr(both['mean_goals_22']):+.3f}")


if __name__ == "__main__":
    main()
