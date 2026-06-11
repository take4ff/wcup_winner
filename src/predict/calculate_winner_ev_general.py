#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calculate_winner_ev_general.py
任意の対戦国とWINNERの18択オッズを指定し、
予測モデルの期待得点（lambda）をロードして期待値（EV）と推奨購入額を算出する。

使い方:
  # 対話モードで起動
  python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Africa"
  
  # 入力用CSVのテンプレートを生成
  python src/predict/calculate_winner_ev_general.py --generate_template data/raw/odds/winner_inputs/winner_template.csv
  
  # 入力用CSVを読み込んで実行
  python src/predict/calculate_winner_ev_general.py --home Mexico --away "South Africa" --odds_csv data/raw/odds/winner_inputs/01_winner_mexico_south_africa.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Dixon-Coles 補正パラメータ (backtest最適値)
RHO = -0.03
MAX_GOALS = 15

# WINNER 18択のキーと日本語名の定義
SELECTION_DEFS = {
    # ホーム勝利
    "H_1-0": {"name": "ホーム 1-0", "type": "H", "goals": (1, 0)},
    "H_2-0": {"name": "ホーム 2-0", "type": "H", "goals": (2, 0)},
    "H_2-1": {"name": "ホーム 2-1", "type": "H", "goals": (2, 1)},
    "H_3-0": {"name": "ホーム 3-0", "type": "H", "goals": (3, 0)},
    "H_3-1": {"name": "ホーム 3-1", "type": "H", "goals": (3, 1)},
    "H_3-2": {"name": "ホーム 3-2", "type": "H", "goals": (3, 2)},
    "H_other": {"name": "ホーム その他(4得点以上)", "type": "H", "goals": None},

    # 引き分け
    "D_0-0": {"name": "引き分け 0-0", "type": "D", "goals": (0, 0)},
    "D_1-1": {"name": "引き分け 1-1", "type": "D", "goals": (1, 1)},
    "D_2-2": {"name": "引き分け 2-2", "type": "D", "goals": (2, 2)},
    "D_other": {"name": "引き分け その他(3得点以上)", "type": "D", "goals": None},

    # アウェイ勝利
    "A_0-1": {"name": "アウェイ 0-1", "type": "A", "goals": (0, 1)},
    "A_0-2": {"name": "アウェイ 0-2", "type": "A", "goals": (0, 2)},
    "A_1-2": {"name": "アウェイ 1-2", "type": "A", "goals": (1, 2)},
    "A_0-3": {"name": "アウェイ 0-3", "type": "A", "goals": (0, 3)},
    "A_1-3": {"name": "アウェイ 1-3", "type": "A", "goals": (1, 3)},
    "A_2-3": {"name": "アウェイ 2-3", "type": "A", "goals": (2, 3)},
    "A_other": {"name": "アウェイ その他(4得点以上)", "type": "A", "goals": None},
}


def score_prob_matrix(lambda_a, lambda_b, rho=RHO):
    """Dixon-Coles補正済み同時確率行列を算出"""
    goals = np.arange(MAX_GOALS + 1)
    pm = np.outer(poisson.pmf(goals, lambda_a), poisson.pmf(goals, lambda_b))
    pm /= pm.sum()
    if abs(rho) > 1e-9:
        pm[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 0.0)
        pm[1, 0] *= max(1.0 + lambda_b * rho, 0.0)
        pm[0, 1] *= max(1.0 + lambda_a * rho, 0.0)
        pm[1, 1] *= max(1.0 - rho, 0.0)
        pm /= pm.sum()
    return pm


def load_match_lambdas(predicted_scores_path, home, away):
    """predicted_scores.csv から対戦国に一致する lambda をロードする"""
    if not os.path.exists(predicted_scores_path):
        print(f"[ERROR] 予測スコアファイルが存在しません: {predicted_scores_path}")
        print("先に予測パイプラインを実行してください。")
        sys.exit(1)

    df = pd.read_csv(predicted_scores_path)
    
    # 通常マッチング
    match = df[(df['home_team'] == home) & (df['away_team'] == away)]
    if len(match) > 0:
        row = match.iloc[0]
        return row['lambda_home'], row['lambda_away'], False

    # ホームアウェイが逆の場合のマッチング
    match_rev = df[(df['home_team'] == away) & (df['away_team'] == home)]
    if len(match_rev) > 0:
        row = match_rev.iloc[0]
        # lambdaを入れ替えて返す
        return row['lambda_away'], row['lambda_home'], True

    return None, None, False


def interactive_get_odds(home, away):
    """対話式に18択のオッズを入力させる"""
    print(f"\n========================================================")
    print(f"オッズ入力モード (対話型): {home} (HOME) vs {away} (AWAY)")
    print("各スコアのオッズ数値を入力し Enter を押してください。")
    print("（未入力のまま Enter を押した項目はデフォルト 1.0 に設定されます）")
    print(f"========================================================")
    
    odds_dict = {}
    
    # グループごとに入力を受ける (ホーム勝利 -> 引き分け -> アウェイ勝利)
    for gtype, label in [("H", f"{home} (HOME) 勝利時のスコア"), 
                         ("D", "引き分け時のスコア"), 
                         ("A", f"{away} (AWAY) 勝利時のスコア")]:
        print(f"\n--- {label} ---")
        for key, defs in SELECTION_DEFS.items():
            if defs["type"] != gtype:
                continue
            
            # 国名表記をホーム/アウェイに置換
            disp_name = defs["name"].replace("ホーム", home).replace("アウェイ", away)
            
            while True:
                val = input(f"  {disp_name:<25} オッズ: ")
                if not val.strip():
                    odds_dict[key] = 1.0
                    break
                try:
                    odds = float(val)
                    if odds <= 0:
                        print("    [Warning] オッズは正の数値を入力してください。")
                        continue
                    odds_dict[key] = odds
                    break
                except ValueError:
                    print("    [Warning] 数値を入力してください。")
                    
    return odds_dict


def generate_template_csv(path):
    """入力用のCSVテンプレートファイルを生成する"""
    records = []
    for key, defs in SELECTION_DEFS.items():
        records.append({
            "selection_key": key,
            "selection_name": defs["name"],
            "odds": ""
        })
    df = pd.DataFrame(records)
    
    # フォルダ作成
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SUCCESS] CSVテンプレートを生成しました: {path}")
    print("このCSVの 'odds' 列に数値を記入して --odds_csv 引数で読み込めます。")


def main():
    parser = argparse.ArgumentParser(description="汎用WINNER期待値計算ツール")
    parser.add_argument("--home", type=str, help="ホームチーム名")
    parser.add_argument("--away", type=str, help="アウェイチーム名")
    parser.add_argument("--bankroll", type=float, default=100000.0, help="想定総資金 (単位: 円)")
    parser.add_argument("--odds_csv", type=str, help="18択オッズを記載したCSVファイル")
    parser.add_argument("--out_csv", type=str, help="結果を出力するCSVファイルのパス")
    parser.add_argument("--match_no", type=str, help="試合の連番（例: 01）")
    parser.add_argument("--generate_template", type=str, help="入力用CSVテンプレートの生成パス（指定時は生成して終了）")
    args = parser.parse_args()

    # テンプレート生成モード
    if args.generate_template:
        generate_template_csv(args.generate_template)
        return

    # 通常実行時の引数チェック
    if not args.home or not args.away:
        print("[ERROR] --home と --away の指定は必須です。(テンプレート生成時を除く)")
        parser.print_help()
        sys.exit(1)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    predicted_scores_path = os.path.join(base_dir, "data/processed/2026/predicted_scores.csv")

    # 1. 予測モデルから lambda のロード
    print(f"Loading expected goals (lambdas) for {args.home} vs {args.away}...")
    lambda_home, lambda_away, is_reversed = load_match_lambdas(predicted_scores_path, args.home, args.away)
    
    if lambda_home is None:
        print(f"[ERROR] 予測スコアデータに該当の対戦が見つかりません: {args.home} vs {args.away}")
        print("スペルを確認するか、対象の試合が含まれているか確認してください。")
        sys.exit(1)
        
    if is_reversed:
        print(f"  (Note: データ内では {args.away} vs {args.home} の順で登録されているため、自動的にホーム/アウェイを反転してロードしました)")
    
    print(f"  Lambdas: {args.home} = {lambda_home:.3f} | {args.away} = {lambda_away:.3f}")

    # 2. オッズデータの取得
    odds_data = {}
    if args.odds_csv:
        if not os.path.exists(args.odds_csv):
            print(f"[ERROR] 指定されたオッズCSVファイルが存在しません: {args.odds_csv}")
            sys.exit(1)
        
        print(f"Loading odds from CSV: {args.odds_csv}")
        df_odds = pd.read_csv(args.odds_csv)
        for _, row in df_odds.iterrows():
            key = row['selection_key']
            val = row['odds']
            if key in SELECTION_DEFS:
                try:
                    odds_data[key] = float(val) if not pd.isna(val) else 1.0
                except ValueError:
                    odds_data[key] = 1.0
    else:
        # 引数がない場合は対話型入力
        odds_data = interactive_get_odds(args.home, args.away)

    # 3. 確率分布の計算
    pm = score_prob_matrix(lambda_home, lambda_away)

    p_home_win = float(np.sum(np.tril(pm, -1)))
    p_draw = float(np.sum(np.diag(pm)))
    p_away_win = float(np.sum(np.triu(pm, 1)))

    # 各スコア確率の分類
    winner_probs = {}
    for key, defs in SELECTION_DEFS.items():
        goals = defs["goals"]
        if goals is not None:
            # 定義済みの固定スコア
            winner_probs[key] = pm[goals[0], goals[1]]
        else:
            # 「その他」の計算
            winner_probs[key] = 0.0  # 後でまとめて計算

    # ホームその他
    defined_h = sum(winner_probs[k] for k, d in SELECTION_DEFS.items() if d["type"] == "H" and d["goals"] is not None)
    winner_probs["H_other"] = max(0.0, p_home_win - defined_h)

    # 引き分けその他
    defined_d = sum(winner_probs[k] for k, d in SELECTION_DEFS.items() if d["type"] == "D" and d["goals"] is not None)
    winner_probs["D_other"] = max(0.0, p_draw - defined_d)

    # アウェイその他
    defined_a = sum(winner_probs[k] for k, d in SELECTION_DEFS.items() if d["type"] == "A" and d["goals"] is not None)
    winner_probs["A_other"] = max(0.0, p_away_win - defined_a)

    # 4. 期待値 (EV) とハーフケリーの算出
    results = []
    for key, defs in SELECTION_DEFS.items():
        prob = winner_probs[key]
        odds = odds_data.get(key, 1.0)
        ev = prob * odds

        if odds > 1.0:
            kelly_f = prob - (1.0 - prob) / (odds - 1.0)
            half_kelly = max(0.0, kelly_f / 2.0)
        else:
            half_kelly = 0.0

        recommended_stake = half_kelly * args.bankroll
        disp_name = defs["name"].replace("ホーム", args.home).replace("アウェイ", args.away)

        results.append({
            "key": key,
            "Selection": disp_name,
            "Odds": odds,
            "Probability(%)": round(prob * 100, 2),
            "Expected Value (EV)": round(ev, 4),
            "Half-Kelly Fraction": round(half_kelly, 5),
            f"Recommended Stake (Bankroll: {int(args.bankroll):,} JPY)": round(recommended_stake, 0)
        })

    results_sorted = sorted(results, key=lambda x: x["Expected Value (EV)"], reverse=True)

    # 5. コンソール出力
    print("\n" + "=" * 90)
    print(f"🇲🇽 {args.home} vs {args.away} 🇿🇦  WINNER 期待値 (EV) ＆ 推奨購入額 (汎用計算結果)")
    print(f"モデル予測: {args.home}勝利 {p_home_win*100:.1f}% / 引き分け {p_draw*100:.1f}% / {args.away}勝利 {p_away_win*100:.1f}%")
    print(f"想定総資金: {int(args.bankroll):,}円")
    print("=" * 90)
    print(f"{'選択肢':<25} {'オッズ':>8} {'予測確率':>10} {'期待値 (EV)':>12} {'ケリー比率':>10} {'推奨購入額':>12}")
    print("-" * 90)

    for r in results_sorted:
        mark = "★" if r["Expected Value (EV)"] >= 1.0 else "  "
        kelly_pct = f"{r['Half-Kelly Fraction'] * 100:.2f}%"
        stake_str = f"{int(r[f'Recommended Stake (Bankroll: {int(args.bankroll):,} JPY)']):,}円" if r[f"Recommended Stake (Bankroll: {int(args.bankroll):,} JPY)"] > 0 else "0円"
        print(f"{mark}{r['Selection']:<25} {r['Odds']:>8.1f} {r['Probability(%)']:>9.2f}% {r['Expected Value (EV)']:>11.3f} {kelly_pct:>10} {stake_str:>12}")
    
    print("=" * 90)
    print("※ 期待値 (EV) >= 1.00 の選択肢に★印を付与しています。")

    # 6. CSVに保存
    if args.out_csv:
        out_path = args.out_csv
    else:
        # デフォルト出力先の設定
        clean_home = args.home.lower().replace(" ", "_")
        clean_away = args.away.lower().replace(" ", "_")
        
        # 番号（match_no）の自動パースまたは明示的指定の適用
        match_no = ""
        if args.match_no:
            match_no = args.match_no.strip()
            if not match_no.endswith("_") and match_no:
                match_no += "_"
        elif args.odds_csv:
            # 入力CSV名から連番を抽出（例: "01_winner_..." -> "01_"）
            base_name = os.path.basename(args.odds_csv)
            import re
            m = re.match(r"^(\d+)_", base_name)
            if m:
                match_no = m.group(1) + "_"
                
        out_dir = os.path.join(base_dir, "data/processed/2026/winner_matches")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{match_no}winner_{clean_home}_{clean_away}_ev.csv")

    df_out = pd.DataFrame(results_sorted)
    df_out = df_out.drop(columns=["key"])
    df_out.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] 計算結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
