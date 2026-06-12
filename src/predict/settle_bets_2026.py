#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
settle_bets_2026.py
購入履歴CSV（data/buy_status.csv）に記録された未確定（pending）のベットについて、
最新の試合結果（data/raw/match/results.csv）と突合して自動的に的中判定を行い、
払戻金および全体の収支（ROI）を計算してCSVを更新する。

使い方:
  python src/predict/settle_bets_2026.py
"""
import os
import sys
import re
import pandas as pd
import numpy as np


def parse_target_score(selection):
    """selection文字列から「1-0」や「0-2」のようなスコア数値を抽出する"""
    match = re.search(r'(\d+)\s*-\s*(\d+)', str(selection))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def is_other_matched(selection, act_h, act_a, home_team, away_team):
    """「その他」と表現される選択肢の的中判定"""
    is_h_win = act_h > act_a
    is_a_win = act_a > act_h
    is_draw = act_h == act_a
    
    sel_str = str(selection).lower()
    home_lower = str(home_team).lower()
    away_lower = str(away_team).lower()
    
    # 1. 引き分けその他（両チーム3得点以上で引き分け）
    if ("draw" in sel_str or "引き分け" in sel_str) and ("その他" in sel_str or "3得点以上" in sel_str):
        return is_draw and act_h >= 3
        
    # 2. ホーム勝利その他（ホームが4得点以上で勝利）
    if ("home" in sel_str or "ホーム" in sel_str or home_lower in sel_str) and ("その他" in sel_str or "4得点" in sel_str):
        return is_h_win and act_h >= 4
        
    # 3. アウェイ勝利その他（アウェイが4得点以上で勝利）
    if ("away" in sel_str or "アウェイ" in sel_str or away_lower in sel_str) and ("その他" in sel_str or "4得点" in sel_str):
        return is_a_win and act_a >= 4
        
    return False


def evaluate_bet(selection, act_h, act_a, home_team, away_team):
    """個別のベットが的中したかどうかを判定する"""
    target_score = parse_target_score(selection)
    
    if target_score is not None:
        # 固定スコアの判定 (例: 0-0, 2-1)
        tgt_h, tgt_a = target_score
        return act_h == tgt_h and act_a == tgt_a
    else:
        # 「その他」スコアの判定
        return is_other_matched(selection, act_h, act_a, home_team, away_team)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    buy_status_path = os.path.join(base_dir, "data/buy_status.csv")
    results_path = os.path.join(base_dir, "data/raw/match/results.csv")

    if not os.path.exists(buy_status_path):
        print(f"[ERROR] 購入履歴CSVが存在しません: {buy_status_path}")
        sys.exit(1)
        
    if not os.path.exists(results_path):
        print(f"[ERROR] 試合結果CSVが存在しません: {results_path}")
        sys.exit(1)

    print("Loading data...")
    df_buy = pd.read_csv(buy_status_path)
    df_res = pd.read_csv(results_path)

    # 日付型に変換
    df_buy['date'] = pd.to_datetime(df_buy['date'])
    df_res['date'] = pd.to_datetime(df_res['date'])

    updated_count = 0
    won_count = 0
    lost_count = 0

    print("\nEvaluating pending bets...")
    for idx, row in df_buy.iterrows():
        # ステータスが pending 以外のものはスキップ
        if str(row['status']).strip().lower() != 'pending':
            continue

        h_team = row['home_team']
        a_team = row['away_team']
        b_date = row['date']

        # 該当の試合を結果から検索 (ホーム/アウェイの反転も考慮)
        # 日付の一致（または前後3日以内の誤差を許容するとより安全ですが、通常は日付一致で検索）
        match = df_res[
            (df_res['date'] == b_date) & 
            (((df_res['home_team'] == h_team) & (df_res['away_team'] == a_team)) |
             ((df_res['home_team'] == a_team) & (df_res['away_team'] == h_team)))
        ]

        if len(match) == 0:
            # 日付なしでチーム名のみで直近の試合を検索してみる (日付が多少ズレている可能性の救済)
            match = df_res[
                (((df_res['home_team'] == h_team) & (df_res['away_team'] == a_team)) |
                 ((df_res['home_team'] == a_team) & (df_res['away_team'] == h_team)))
            ].sort_values('date')
            
            # W杯期間中(2026年6月)の試合に絞る
            match = match[(match['date'] >= '2026-06-01') & (match['date'] <= '2026-07-31')]

        # 手入力の match_result 列（購入時のホーム/アウェイ視点, 例: "2-0"）があれば優先利用
        manual_score = parse_target_score(row['match_result']) if 'match_result' in df_buy.columns else None

        act_home_score = act_away_score = None
        if manual_score is not None:
            act_home_score, act_away_score = manual_score
        elif len(match) > 0:
            match_row = match.iloc[0]
            act_h = match_row['home_score']
            act_a = match_row['away_score']

            # スコアが NaN の場合は試合前のためスキップ
            if pd.isna(act_h) or pd.isna(act_a):
                continue

            # 実際の試合においてホーム/アウェイが逆転して記録されているか
            is_reversed = match_row['home_team'] == a_team

            # 判定用にホーム/アウェイの実際の得点を整理
            # (購入側視点でのホーム/アウェイ得点にする)
            if is_reversed:
                act_home_score = int(act_a)
                act_away_score = int(act_h)
            else:
                act_home_score = int(act_h)
                act_away_score = int(act_a)

        if act_home_score is not None:

            # 的中判定
            is_won = evaluate_bet(row['selection'], act_home_score, act_away_score, h_team, a_team)
            odds = float(row['odds'])
            buy_amt = float(row['buy_amount_yen'])

            if is_won:
                result_amt = buy_amt * odds
                status = 'won'
                won_count += 1
                result_str = f"的中! 🎉 (スコア: {act_home_score}-{act_away_score} | 払戻金: {result_amt:,.0f}円)"
            else:
                result_amt = 0.0
                status = 'lost'
                lost_count += 1
                result_str = f"不的中 (実際のスコア: {act_home_score}-{act_away_score})"

            df_buy.at[idx, 'result_amount_yen'] = result_amt
            df_buy.at[idx, 'status'] = status
            if 'match_result' in df_buy.columns:
                df_buy.at[idx, 'match_result'] = f"{act_home_score}-{act_away_score}"
            updated_count += 1

            disp_date = b_date.strftime('%Y-%m-%d')
            print(f"  [{disp_date}] {h_team} vs {a_team} (賭け: {row['selection']}) -> {result_str}")

    # 結果の保存
    if updated_count > 0:
        # 日付を文字列に戻して保存
        df_buy['date'] = df_buy['date'].dt.strftime('%Y-%m-%d')
        # float型を整数表記にする (NaNでないもの)
        df_buy['result_amount_yen'] = df_buy['result_amount_yen'].apply(lambda x: int(x) if not pd.isna(x) else "")
        df_buy.to_csv(buy_status_path, index=False)
        print(f"\n[SUCCESS] {updated_count} 件のベット結果を確定し、{buy_status_path} を更新しました。")
    else:
        print("\n確定可能な新規の試合結果はありませんでした。")

    # 全体集計の出力
    df_summary = pd.read_csv(buy_status_path)
    total_invested = df_summary['buy_amount_yen'].sum()
    
    # 確定済みの回収金額を取得 (空文字やNaNを0にする)
    df_summary['result_amount_yen'] = pd.to_numeric(df_summary['result_amount_yen'], errors='coerce').fillna(0)
    total_returned = df_summary['result_amount_yen'].sum()
    
    net_profit = total_returned - total_invested
    roi = (total_returned / total_invested * 100) if total_invested > 0 else 0.0

    won_total = len(df_summary[df_summary['status'] == 'won'])
    lost_total = len(df_summary[df_summary['status'] == 'lost'])
    pending_total = len(df_summary[df_summary['status'] == 'pending'])

    print("\n" + "=" * 50)
    print("🏆 WINNER 通算運用成績")
    print("=" * 50)
    print(f"  総投資額  : {total_invested:,.0f} 円")
    print(f"  総回収額  : {total_returned:,.0f} 円")
    print(f"  トータル収支: {net_profit:+,.0f} 円")
    print(f"  回収率 (ROI): {roi:.2f} %")
    print("-" * 50)
    print(f"  確定済みベット数: {won_total + lost_total} (的中: {won_total} / 不的中: {lost_total})")
    print(f"  未確定ベット数  : {pending_total}")
    print("=" * 50)


if __name__ == "__main__":
    main()
