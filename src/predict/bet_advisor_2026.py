"""
bet_advisor_2026.py
2026年W杯グループステージ全試合の期待値（EV）を計算し、
DNB/DC戦略・ケリー基準に基づいてベッティング推奨を出力する。

使い方:
  python src/bet_advisor_2026.py [--ev_thresh 1.05] [--kelly half] [--bankroll 100]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np


def dnb_dc_ev(p_home, p_draw, p_away, odds_home, odds_draw, odds_away):
    """
    3つの賭けタイプのEVを返す:
      - 1x2 (通常: home/draw/away)
      - DNB home (引き分けなら返金)
      - DNB away (引き分けなら返金)
      - DC 1X (home or draw)
      - DC X2 (draw or away)
    """
    results = {}

    # 1X2 通常
    results['H'] = {'type': '1X2 Home',    'ev': p_home * odds_home - 1, 'odds': odds_home, 'prob': p_home}
    results['D'] = {'type': '1X2 Draw',    'ev': p_draw * odds_draw - 1, 'odds': odds_draw, 'prob': p_draw}
    results['A'] = {'type': '1X2 Away',    'ev': p_away * odds_away - 1, 'odds': odds_away, 'prob': p_away}

    # DNB (Draw No Bet): 引き分け時返金 → 実質odds = 1 + (home_odds-1) / (1-p_draw)
    # EV_DNB_home = p_home * dnb_odds - 1  (ただし引き分け時は0)
    if p_home + p_away > 0:
        dnb_home_odds = (odds_home - 1) / (1 - 1/odds_draw) + 1 if odds_draw > 1 else odds_home
        dnb_home_odds = max(1.0, (1 + (odds_home - 1) * (p_home / (p_home + p_away))) / (p_home / (p_home + p_away)))
        # シンプルな計算: DNB odds = prob_win / (prob_win + prob_loss) に基づく
        # 実際のDNBオッズ = 1 / (p_home / (p_home + p_away))
        dnb_h_eff_odds = (p_home + p_away) / p_home
        results['DNB_H'] = {
            'type': 'DNB Home',
            'ev': p_home * dnb_h_eff_odds - (p_home + p_away),
            'odds': round(dnb_h_eff_odds, 3),
            'prob': p_home / (p_home + p_away)
        }

        dnb_a_eff_odds = (p_home + p_away) / p_away
        results['DNB_A'] = {
            'type': 'DNB Away',
            'ev': p_away * dnb_a_eff_odds - (p_home + p_away),
            'odds': round(dnb_a_eff_odds, 3),
            'prob': p_away / (p_home + p_away)
        }

    # Double Chance (DC)
    dc_1x_odds = 1 / (p_home + p_draw) if (p_home + p_draw) > 0 else 999
    dc_x2_odds = 1 / (p_draw + p_away) if (p_draw + p_away) > 0 else 999
    # ブックメーカーDCオッズ (簡易推定: 2つの勝敗オッズから計算)
    bm_dc_1x = 1 / (1/odds_home + 1/odds_draw)
    bm_dc_x2 = 1 / (1/odds_draw + 1/odds_away)
    results['DC_1X'] = {'type': 'DC 1X',  'ev': (p_home + p_draw) * bm_dc_1x - 1, 'odds': round(bm_dc_1x, 3), 'prob': p_home + p_draw}
    results['DC_X2'] = {'type': 'DC X2',  'ev': (p_draw + p_away) * bm_dc_x2 - 1, 'odds': round(bm_dc_x2, 3), 'prob': p_draw + p_away}

    return results


def kelly_fraction(prob, odds, mode='half', cap=0.10):
    """ケリー基準によるベット比率を計算"""
    b = odds - 1
    q = 1 - prob
    f = (prob * b - q) / b
    f = max(0.0, f)
    if mode == 'full':
        return min(f, cap * 2)
    elif mode == 'half':
        return min(f / 2, cap)
    elif mode == 'quarter':
        return min(f / 4, cap / 2)
    return min(f / 2, cap)


def main():
    parser = argparse.ArgumentParser(description='2026 W杯 ベッティングアドバイザー')
    parser.add_argument('--ev_thresh', type=float, default=1.05, help='EV閾値 (default: 1.05)')
    parser.add_argument('--kelly',     type=str,   default='half', choices=['full','half','quarter'], help='ケリー倍率')
    parser.add_argument('--bankroll',  type=float, default=100.0, help='初期資金 (units)')
    parser.add_argument('--top',       type=int,   default=30,    help='出力する推奨賭け数の上限')
    parser.add_argument('--max_ev',    type=float, default=2.0,
                        help='EVサニティ上限。これを超えるEVはデータ異常（オッズの並び違い等）の疑いが強いため警告して除外 (default: 2.0)')
    parser.add_argument('--market_blend', type=float, default=0.5,
                        help='モデル確率を市場インプライド確率(マージン除去済み)へシュリンクする比率。'
                             '0=モデルのみ, 1=市場のみ。EV過大評価とケリー賭け金の暴れを抑制 '
                             '(default: 0.5, walkforwardのW杯オッズ評価で0.4-0.7が最良域)')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_probs_path = os.path.join(base_dir, "data/processed/2026/predicted_scores.csv")
    odds_path        = os.path.join(base_dir, "data/raw/odds/odds_groups_2026.csv")
    output_path      = os.path.join(base_dir, "data/processed/2026/bet_recommendations.csv")

    print("Loading model predictions and bookmaker odds...")
    df_model = pd.read_csv(model_probs_path)
    df_odds  = pd.read_csv(odds_path)

    # マージキー生成（対称）
    def make_key(h, a): return f"{min(h,a)}_{max(h,a)}"
    df_model['key'] = df_model.apply(lambda r: make_key(r['home_team'], r['away_team']), axis=1)
    df_odds['key']  = df_odds.apply( lambda r: make_key(r['home_team'], r['away_team']), axis=1)

    df = df_model.merge(
        df_odds[['key','home_team','odds_home','odds_draw','odds_away','source']]
            .rename(columns={'home_team': 'odds_home_team'}),
        on='key', how='inner')

    # オッズCSVとモデル側で home/away の並びが逆の試合はオッズを入れ替える
    flipped = df['odds_home_team'] != df['home_team']
    df.loc[flipped, ['odds_home', 'odds_away']] = df.loc[flipped, ['odds_away', 'odds_home']].values

    print(f"  Merged: {len(df)} matches (home/away order corrected: {flipped.sum()})\n")

    recommendations = []

    for _, row in df.iterrows():
        home, away, grp = row['home_team'], row['away_team'], row['group']
        p_h = row['p_home_win'] / 100
        p_d = row['p_draw']     / 100
        p_a = row['p_away_win'] / 100
        o_h, o_d, o_a = row['odds_home'], row['odds_draw'], row['odds_away']
        src = row.get('source', 'estimated')

        # 市場インプライド確率へのシュリンク（実オッズの試合のみ。ブックメーカーマージンは正規化で除去）
        if args.market_blend > 0 and src == 'real':
            m_h, m_d, m_a = 1 / o_h, 1 / o_d, 1 / o_a
            overround = m_h + m_d + m_a
            m_h, m_d, m_a = m_h / overround, m_d / overround, m_a / overround
            w = args.market_blend
            p_h = (1 - w) * p_h + w * m_h
            p_d = (1 - w) * p_d + w * m_d
            p_a = (1 - w) * p_a + w * m_a

        evs = dnb_dc_ev(p_h, p_d, p_a, o_h, o_d, o_a)

        for bet_key, ev_info in evs.items():
            ev    = ev_info['ev']
            odds  = ev_info['odds']
            prob  = ev_info['prob']
            btype = ev_info['type']

            if ev > (args.ev_thresh - 1):  # EV > threshold-1 (evは収益率なので1.05閾値なら0.05)
                if (1 + ev) > args.max_ev:
                    print(f"[WARNING] EVサニティ上限超え → 除外: {home} vs {away} "
                          f"{btype} odds={odds:.2f} EV={1+ev:.3f} "
                          f"(オッズとモデルの対応関係を確認してください)")
                    continue
                kelly_f = kelly_fraction(prob, odds, mode=args.kelly)
                stake   = round(args.bankroll * kelly_f, 2)
                recommendations.append({
                    'group':      grp,
                    'home_team':  home,
                    'away_team':  away,
                    'bet_type':   btype,
                    'model_prob': f"{prob*100:.1f}%",
                    'odds':       odds,
                    'EV':         round(1 + ev, 4),
                    'kelly_frac': f"{kelly_f*100:.1f}%",
                    'stake':      stake,
                    'odds_source': src,
                })

    df_rec = pd.DataFrame(recommendations)
    if len(df_rec) == 0:
        print("推奨ベットが見つかりませんでした。EV閾値を下げてみてください。")
        return

    df_rec = df_rec.sort_values('EV', ascending=False).reset_index(drop=True)

    print(f"{'='*100}")
    print(f"🎯 2026 W杯 ベッティング推奨 ({args.kelly}-Kelly | 初期資金: {args.bankroll} units | EV閾値: {args.ev_thresh})")
    print(f"{'='*100}")
    print(f"{'GRP':<4} {'Home':<20} {'Away':<20} {'Bet Type':<12} {'Prob':>6} {'Odds':>6} {'EV':>6} {'Kelly':>6} {'Stake':>7} {'Src':<10}")
    print(f"{'-'*100}")

    for _, r in df_rec.head(args.top).iterrows():
        src_label = '★Real' if r['odds_source'] == 'real' else 'Est.'
        print(f"{r['group']:<4} {r['home_team']:<20} {r['away_team']:<20} "
              f"{r['bet_type']:<12} {r['model_prob']:>6} {r['odds']:>6.2f} "
              f"{r['EV']:>6.3f} {r['kelly_frac']:>6} {r['stake']:>7.2f} {src_label:<10}")

    print(f"{'='*100}")
    total_stake = df_rec.head(args.top)['stake'].sum()
    print(f"\n推奨合計ベット数: {min(len(df_rec), args.top)} / 総賭け金: {total_stake:.2f} units")
    print(f"EV>1.05のベット数: {len(df_rec[df_rec['EV'] >= 1.05])}")

    # 実際のオッズがある試合のみ抜粋
    real_bets = df_rec[df_rec['odds_source'] == 'real']
    if len(real_bets) > 0:
        print(f"\n{'='*60}")
        print("★ 実際のオッズによる推奨ベット（最優先）")
        print(f"{'='*60}")
        for _, r in real_bets.iterrows():
            print(f"  {r['group']} | {r['home_team']} vs {r['away_team']}")
            print(f"  → {r['bet_type']} | オッズ: {r['odds']:.2f} | EV: {r['EV']:.3f} | "
                  f"ケリー: {r['kelly_frac']} | 賭け金: {r['stake']} units")
        print()

    df_rec.to_csv(output_path, index=False)
    print(f"推奨リストを保存: {output_path}")

    # === 日本関連の賭け ===
    japan_bets = df_rec[(df_rec['home_team'] == 'Japan') | (df_rec['away_team'] == 'Japan')]
    if len(japan_bets) > 0:
        print(f"\n{'='*60}")
        print("🇯🇵 日本代表関連の推奨ベット")
        print(f"{'='*60}")
        for _, r in japan_bets.iterrows():
            print(f"  {r['home_team']} vs {r['away_team']} | {r['bet_type']}")
            print(f"  → オッズ: {r['odds']:.2f} | EV: {r['EV']:.3f} | "
                  f"ケリー: {r['kelly_frac']} | 推奨賭け金: {r['stake']} units")
        print()


if __name__ == "__main__":
    main()
