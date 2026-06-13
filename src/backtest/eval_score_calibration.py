#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_score_calibration.py - 「正確スコア確率」の較正検証（WINNERスコア市場の前提テスト）

1X2(勝/分/負)の較正は ECE 0.0123 と検証済みだが、WINNER で実際に賭けるのは
Dixon-Coles 行列由来の「正確スコア / その他」である。本スクリプトはその確率が
実頻度と一致するか（較正されているか）を、リークなしの walkforward 予測で直接測る。

検証する3つの観点:
  A. スコアセル別の較正    : モデルの平均 P(i-j) vs 実頻度（1-0, 2-1 等は WINNER の選択肢そのもの）
  B. スコア市場 ECE        : 全(試合×セル)予測確率をビン分割した期待較正誤差（1X2 ECE のスコア版）
  C. 総得点分布の較正      : P(総得点=k) vs 実頻度（「その他(4得点以上)」= 高得点側の前提を直接テスト）

λは本番に近い3モデルアンサンブル (Poisson+LGBM+LGBM_cat)/3、ρ=-0.09。

使い方:
  python src/backtest/eval_score_calibration.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RHO, MAXG = -0.09, 10
GRID = 6  # 0..5 を個別、6 を「6点以上」の集約として扱う上限


def dc_matrix(lh, la, rho=RHO):
    g = np.arange(MAXG + 1)
    pm = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    pm /= pm.sum()
    pm[0, 0] *= max(1.0 - lh * la * rho, 0.0)
    pm[1, 0] *= max(1.0 + la * rho, 0.0)
    pm[0, 1] *= max(1.0 + lh * rho, 0.0)
    pm[1, 1] *= max(1.0 - rho, 0.0)
    pm /= pm.sum()
    return pm


def main():
    preds = pd.read_csv(os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv"))
    lam_h = (preds['lam_p_h'] + preds['lam_l_h'] + preds['lam_c_h']).values / 3.0
    lam_a = (preds['lam_p_a'] + preds['lam_l_a'] + preds['lam_c_a']).values / 3.0
    hs = preds['home_score'].clip(upper=GRID).astype(int).values
    as_ = preds['away_score'].clip(upper=GRID).astype(int).values
    n = len(preds)
    print(f"評価試合: {n}（リークなし walkforward, λ=(P+L+cat)/3, ρ={RHO}）\n")

    # 各試合の縮約スコア行列 (0..GRID、GRID行/列は「以上」を集約)
    pred_cell = np.zeros((n, GRID + 1, GRID + 1))
    for k in range(n):
        pm = dc_matrix(lam_h[k], lam_a[k])  # 11x11
        red = pm[:GRID + 1, :GRID + 1].copy()
        red[GRID, :] += pm[GRID + 1:, :GRID + 1].sum(axis=0)
        red[:, GRID] += pm[:GRID + 1, GRID + 1:].sum(axis=1)
        red[GRID, GRID] += pm[GRID + 1:, GRID + 1:].sum()
        pred_cell[k] = red

    # 実測の one-hot
    obs_cell = np.zeros((n, GRID + 1, GRID + 1))
    obs_cell[np.arange(n), hs, as_] = 1.0

    # ===== A. スコアセル別の較正（上位頻度のみ表示） =====
    pred_mean = pred_cell.mean(axis=0)
    obs_freq = obs_cell.mean(axis=0)
    print("=== A. スコアセル別の較正（モデル平均確率 vs 実頻度）===")
    print(f"{'スコア':>7}{'予測%':>9}{'実測%':>9}{'実測/予測':>10}{'件数':>7}")
    cells = [(i, j) for i in range(GRID + 1) for j in range(GRID + 1)]
    cells.sort(key=lambda c: obs_freq[c], reverse=True)
    wce = 0.0
    for i, j in cells:
        if obs_freq[i, j] < 0.005:
            continue
        cnt = int(obs_cell[:, i, j].sum())
        ratio = obs_freq[i, j] / pred_mean[i, j] if pred_mean[i, j] > 0 else float('nan')
        lbl = f"{i}-{j}" + ("+" if i == GRID or j == GRID else "")
        print(f"{lbl:>7}{pred_mean[i,j]*100:>8.2f}%{obs_freq[i,j]*100:>8.2f}%{ratio:>10.2f}{cnt:>7}")
        wce += abs(obs_freq[i, j] - pred_mean[i, j])
    print(f"  → 上位セルの絶対較正誤差合計（加重前）: {wce*100:.2f}ポイント")

    # ===== B. スコア市場 ECE（全セル予測のビン較正）=====
    p_flat = pred_cell.ravel()
    o_flat = obs_cell.ravel()
    bins = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25, 1.01])
    print("\n=== B. スコア市場 ECE（全 試合×49セル 予測のビン別較正）===")
    print(f"{'予測確率帯':>14}{'平均予測%':>10}{'実頻度%':>9}{'件数':>9}")
    ece = 0.0
    for b in range(len(bins) - 1):
        m = (p_flat >= bins[b]) & (p_flat < bins[b + 1])
        if m.sum() == 0:
            continue
        mp, mo = p_flat[m].mean(), o_flat[m].mean()
        ece += (m.sum() / len(p_flat)) * abs(mp - mo)
        print(f"{f'[{bins[b]:.3f},{bins[b+1]:.2f})':>14}{mp*100:>9.2f}%{mo*100:>8.2f}%{m.sum():>9}")
    print(f"  → スコア市場 ECE = {ece:.4f}（参考: 1X2のECEは0.0123）")

    # ===== C. 総得点分布の較正（「その他=4得点以上」の前提）=====
    tot = np.arange(2 * GRID + 1)
    pred_tot = np.array([pred_cell[:, i, j].mean()
                         for k in tot for i in range(GRID + 1) for j in range(GRID + 1) if i + j == k])
    pred_total = np.zeros(2 * GRID + 1)
    for i in range(GRID + 1):
        for j in range(GRID + 1):
            pred_total[i + j] += pred_cell[:, i, j].mean()
    real_total = (hs + as_)
    print("\n=== C. 総得点分布の較正（P(総得点=k) vs 実頻度）===")
    print(f"{'総得点':>7}{'予測%':>9}{'実測%':>9}")
    for k in range(0, 2 * GRID + 1):
        of = (real_total == k).mean()
        if pred_total[k] < 0.002 and of < 0.002:
            continue
        lbl = f"{k}+" if k == 2 * GRID else str(k)
        print(f"{lbl:>7}{pred_total[k]*100:>8.2f}%{of*100:>8.2f}%")
    # 「4得点以上」= WINNER「その他」が主に対応する高得点側
    p_4plus = pred_total[4:].sum()
    o_4plus = (real_total >= 4).mean()
    print(f"  → 総得点4以上: 予測 {p_4plus*100:.2f}% / 実測 {o_4plus*100:.2f}%"
          f"（実測/予測 {o_4plus/p_4plus:.2f}）  ← WINNER『その他(4得点以上)』に対応")

    # ===== スコア市場 Log Loss（多クラス）=====
    p_actual = pred_cell[np.arange(n), hs, as_]
    sll = -np.mean(np.log(np.clip(p_actual, 1e-15, 1)))
    base = obs_cell.mean(axis=0)  # 経験周辺分布
    p_base = base[hs, as_]
    sll_base = -np.mean(np.log(np.clip(p_base, 1e-15, 1)))
    print(f"\nスコア市場 Log Loss: モデル {sll:.4f} / 経験周辺分布ベースライン {sll_base:.4f}")


if __name__ == "__main__":
    main()
