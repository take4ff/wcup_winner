#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calculate_outcome_ev.py - WINNER「チーム成績予想」市場の期待値計算

指定チームの最終成績（GS敗退は勝ち点別 / ベスト32 / ベスト16 / ベスト8 /
4位 / 3位 / 準優勝 / 優勝）の確率をモンテカルロで推定し、
WINNERのオッズと突き合わせてEVとハーフケリーを出す。

使い方:
  python src/predict/calculate_outcome_ev.py --team Japan \
      --odds_csv data/raw/odds/winner_inputs/outcome_japan.csv [--sims 20000]
"""
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator_2026 import (  # noqa: E402
    GROUPS_2026, R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS,
    load_team_features, precompute_all_lambdas, predict_match,
    simulate_knockout_match, assign_third_places,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def simulate_with_points(lambda_cache, focus_team):
    """1大会をシミュレートし、focus_team の (最終成績キー, GS勝ち点) を返す"""
    group_top2, all_third, focus_pts = {}, [], 0

    for grp, teams in GROUPS_2026.items():
        pts, gf, ga, h2h = defaultdict(int), defaultdict(int), defaultdict(int), {}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                s1, s2 = predict_match(t1, t2, lambda_cache)
                gf[t1] += s1; ga[t1] += s2; gf[t2] += s2; ga[t2] += s1
                if s1 > s2: pts[t1] += 3; h2h[(t1, t2)] = True
                elif s1 < s2: pts[t2] += 3; h2h[(t2, t1)] = True
                else: pts[t1] += 1; pts[t2] += 1
        standings = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
        for k in range(len(standings) - 1):
            a, b = standings[k], standings[k + 1]
            if (pts[a], gf[a] - ga[a], gf[a]) == (pts[b], gf[b] - ga[b], gf[b]) and h2h.get((b, a)):
                standings[k], standings[k + 1] = b, a
        group_top2[grp] = standings[:2]
        third = standings[2]
        all_third.append((grp, third, pts[third], gf[third] - ga[third], gf[third]))
        if focus_team in teams:
            focus_pts = pts[focus_team]

    thirds_sorted = sorted(all_third, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [(t[0], t[1]) for t in thirds_sorted[:8]]
    third_assignment = assign_third_places(best_thirds)

    def resolve(slot, no):
        kind, val = slot
        if kind == 'W': return group_top2[val][0]
        if kind == 'R': return group_top2[val][1]
        return third_assignment[no]

    winners, in_r32 = {}, set()
    for no, sa, sb in R32_BRACKET:
        ta, tb = resolve(sa, no), resolve(sb, no)
        in_r32.update([ta, tb])
        winners[no], _ = simulate_knockout_match(ta, tb, lambda_cache)
    if focus_team not in in_r32:
        return f'GS{focus_pts}', focus_pts

    stage = 'R32'
    r16, qf, sf = set(), set(), set()
    for no, (ma, mb) in enumerate(R16_PAIRS, start=89):
        ta, tb = winners[ma], winners[mb]
        r16.update([ta, tb])
        winners[no], _ = simulate_knockout_match(ta, tb, lambda_cache)
    for no, (ma, mb) in enumerate(QF_PAIRS, start=97):
        ta, tb = winners[ma], winners[mb]
        qf.update([ta, tb])
        winners[no], _ = simulate_knockout_match(ta, tb, lambda_cache)
    finalists, losers = [], []
    for no, (ma, mb) in enumerate(SF_PAIRS, start=101):
        ta, tb = winners[ma], winners[mb]
        sf.update([ta, tb])
        win, loss = simulate_knockout_match(ta, tb, lambda_cache)
        winners[no] = win
        finalists.append(win); losers.append(loss)
    third_w, _ = simulate_knockout_match(losers[0], losers[1], lambda_cache)
    champ, runner = simulate_knockout_match(finalists[0], finalists[1], lambda_cache)

    if focus_team == champ: return 'Champion', focus_pts
    if focus_team == runner: return '2nd', focus_pts
    if focus_team in finalists: return '2nd', focus_pts  # 保険（到達不能のはず）
    if focus_team in losers:
        return ('3rd' if focus_team == third_w else '4th'), focus_pts
    if focus_team in sf: return 'QF', focus_pts      # SF進出=ベスト8敗退ではない…QF敗退=ベスト8
    if focus_team in qf: return 'QF', focus_pts
    if focus_team in r16: return 'R16', focus_pts
    return 'R32', focus_pts


def main():
    parser = argparse.ArgumentParser(description='WINNER チーム成績予想のEV計算')
    parser.add_argument('--team', type=str, required=True)
    parser.add_argument('--odds_csv', type=str, required=True)
    parser.add_argument('--sims', type=int, default=20000)
    parser.add_argument('--bankroll', type=float, default=100000.0)
    args = parser.parse_args()

    print("Loading models and team features...")
    df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/features.csv"))
    df['date'] = pd.to_datetime(df['date'])
    md = os.path.join(BASE_DIR, "models")
    model_poisson = joblib.load(os.path.join(md, "poisson_model.joblib"))
    model_lgbm = joblib.load(os.path.join(md, "lgbm_model.joblib"))
    feature_cols = joblib.load(os.path.join(md, "feature_cols.joblib"))
    cat_path = os.path.join(md, "lgbm_cat_model.joblib")
    model_cat = joblib.load(cat_path) if os.path.exists(cat_path) else None
    cats = (joblib.load(os.path.join(md, "team_categories.joblib"))
            if os.path.exists(os.path.join(md, "team_categories.joblib")) else None)

    team_feats = load_team_features(df)
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols,
                                          model_cat, cats)

    print(f"Simulating {args.sims} tournaments for {args.team}...")
    counts = defaultdict(int)
    for s in range(args.sims):
        key, _ = simulate_with_points(lambda_cache, args.team)
        counts[key] += 1
        if (s + 1) % 5000 == 0:
            print(f"  {s + 1}/{args.sims}")

    odds_df = pd.read_csv(args.odds_csv)
    results = []
    for r in odds_df.itertuples():
        p = counts.get(r.selection_key, 0) / args.sims
        ev = p * r.odds
        b = r.odds - 1
        kelly = max(0.0, (p * b - (1 - p)) / b) / 2 if b > 0 else 0.0
        results.append({'Selection': r.selection_name, 'Odds': r.odds,
                        'Probability(%)': round(p * 100, 2), 'EV': round(ev, 4),
                        'Half-Kelly': round(kelly, 5),
                        'Stake(JPY)': int(kelly * args.bankroll)})
    out = pd.DataFrame(results).sort_values('EV', ascending=False)

    print(f"\n=== {args.team} 成績予想 EV (sims={args.sims}) ===")
    print(out.to_string(index=False))
    total_p = sum(counts.values()) / args.sims
    out_path = os.path.join(BASE_DIR, f"data/processed/2026/winner_matches/outcome_{args.team.lower()}_ev.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
