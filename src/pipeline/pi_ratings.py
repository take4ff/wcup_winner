#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pi_ratings.py - Constantinou & Fenton (2013) の pi-ratings

各チームに「ホームレーティング R_H」と「アウェイレーティング R_A」を持たせ、
試合ごとに「予測得失点差と実際の得失点差の乖離」を逓減関数で重み付けして更新する。
ベンチマーク研究（2023 Soccer Prediction Challenge 等）で 1X2 の上位特徴量とされる。

本実装は試合結果のみから計算でき（MLモデル非依存）、各試合の「試合前」予測得失点差
（home視点 ĝ_D）をリークなしで返すため、walkforward 評価にそのまま使える。

参考: Constantinou & Fenton (2013), JQAS. 推奨値 λ=0.06, γ=0.6, b=10, c=3。
"""
import numpy as np
import pandas as pd

LAMBDA = 0.06   # 学習率（新しい結果がレーティングを更新する度合い）
GAMMA = 0.6     # ホーム↔アウェイ レーティングの相互更新率
B = 10.0        # 得失点差変換の底
C = 3.0         # 得失点差変換のスケール（逓減）


def _rating_to_gd(r):
    """レーティング r → 期待得失点差。ψ(gd)=C*log_b(1+gd) の逆関数。"""
    return np.sign(r) * (B ** (np.abs(r) / C) - 1.0)


def _gd_to_psi(e):
    """得失点差の乖離 e（絶対値）→ 逓減重み ψ(e)=C*log_b(1+e)。"""
    return C * np.log10(1.0 + e) / np.log10(B)


def compute_pi_predicted_gd(df, lam=LAMBDA, gamma=GAMMA, neutral=None):
    """日付昇順の試合DataFrame（home_team, away_team, home_score, away_score）に対し、
    各試合の「試合前」予測得失点差 ĝ_D（home視点）を配列で返す（リークなし）。

    df は date でソート済みであること。NaNスコアの試合はレーティング更新をスキップし、
    予測値のみ算出する（将来試合の予測に使える）。

    neutral: 中立地フラグ（bool配列）。指定時、中立試合ではホーム/アウェイ別レーティングの
    平均（=ホームアドバンテージを除いた実力）で予測し、更新も home/away 両レーティングに
    対称適用する（W杯など中立開催での誤ったホーム補正を防ぐ）。
    """
    rh = {}  # team -> home rating
    ra = {}  # team -> away rating
    home = df['home_team'].values
    away = df['away_team'].values
    hs = df['home_score'].values
    as_ = df['away_score'].values
    neu = (np.zeros(len(df), dtype=bool) if neutral is None
           else np.asarray(neutral).astype(bool))

    pred_gd = np.empty(len(df))
    for i in range(len(df)):
        h, a = home[i], away[i]
        r_hh, r_ha = rh.get(h, 0.0), ra.get(h, 0.0)
        r_ah, r_aa = rh.get(a, 0.0), ra.get(a, 0.0)
        if neu[i]:
            # 中立地: 各チームの home/away 平均（ホームアドバンテージ抜きの実力）で予測
            ghat = _rating_to_gd((r_hh + r_ha) / 2.0) - _rating_to_gd((r_ah + r_aa) / 2.0)
        else:
            ghat = _rating_to_gd(r_hh) - _rating_to_gd(r_aa)
        pred_gd[i] = ghat

        if np.isnan(hs[i]) or np.isnan(as_[i]):
            continue  # 結果未確定の試合は更新しない

        gd = hs[i] - as_[i]               # 実際の得失点差（home視点）
        e = abs(gd - ghat)               # 予測との乖離
        psi = _gd_to_psi(e)
        sign_h = 1.0 if (gd - ghat) > 0 else -1.0
        delta = psi * lam * sign_h

        if neu[i]:
            # 中立地: home/away 両レーティングへ対称に適用（ホーム偏重を避ける）
            rh[h] = r_hh + delta
            ra[h] = r_ha + delta
            rh[a] = r_ah - delta
            ra[a] = r_aa - delta
        else:
            # home チーム: ホームレーティングを更新、アウェイレーティングを γ で相互更新
            rh[h] = r_hh + delta
            ra[h] = r_ha + delta * gamma
            # away チーム: アウェイレーティングを逆方向に更新、ホームレーティングを相互更新
            ra[a] = r_aa - delta
            rh[a] = r_ah - delta * gamma

    return pred_gd


if __name__ == "__main__":
    # 簡易動作確認
    import os
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    df = pd.read_csv(os.path.join(base, "data/processed/features.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    gd = compute_pi_predicted_gd(df)
    df['pi_gd'] = gd
    played = df.dropna(subset=['home_score', 'away_score'])
    corr = np.corrcoef(played['pi_gd'], played['home_score'] - played['away_score'])[0, 1]
    print(f"matches={len(df)}  pi_gd vs 実得失点差の相関={corr:+.3f}")
    print(played[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'pi_gd']].tail(5).to_string())
