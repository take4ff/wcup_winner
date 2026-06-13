#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_pi_ratings.py - pi-ratings を λ アンサンブルの追加成分として評価する（リークなし）

walkforward.py --mode probs が生成した walkforward_predictions.csv（2018-2026の評価試合・
各成分λ・分類器確率・実結果）に、pi-ratings の試合前予測得失点差をマージし、
λペアへ変換して現行アンサンブル(P+L)と Log Loss / Brier を比較する。

MLモデルの再学習は不要（pi-ratings は試合結果のみから計算）。Eloと差し替えるのではなく
「追加成分として効くか」を、既存の評価条件（ρ=-0.09, cls_blend=0.25, cls=(LGBM+XGB)/2）で検証する。

使い方:
  python src/backtest/eval_pi_ratings.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(BASE_DIR, "src/pipeline"))
from pi_ratings import compute_pi_predicted_gd  # noqa: E402

RHO, W_CLS, MAX_GOALS = -0.09, 0.25, 10


def dc_outcome_probs(lh_arr, la_arr, rho=RHO):
    """λ配列 → [pH,pD,pA] 行列（Dixon-Coles補正つき）"""
    goals = np.arange(MAX_GOALS + 1)
    out = np.empty((len(lh_arr), 3))
    for i, (lh, la) in enumerate(zip(lh_arr, la_arr)):
        pm = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
        pm /= pm.sum()
        pm[0, 0] *= max(1.0 - lh * la * rho, 0.0)
        pm[1, 0] *= max(1.0 + la * rho, 0.0)
        pm[0, 1] *= max(1.0 + lh * rho, 0.0)
        pm[1, 1] *= max(1.0 - rho, 0.0)
        pm /= pm.sum()
        out[i] = [np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))]
    return out


def logloss(p, y):
    eps = 1e-15
    p = np.clip(p / p.sum(axis=1, keepdims=True), eps, 1 - eps)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def brier(p, y):
    p = p / p.sum(axis=1, keepdims=True)
    return np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))


def main():
    # 1) 全試合の pi-ratings 試合前予測得失点差（リークなし）
    feats = pd.read_csv(os.path.join(BASE_DIR, "data/processed/features.csv"))
    feats['date'] = pd.to_datetime(feats['date'])
    feats = feats.sort_values('date').reset_index(drop=True)
    feats['pi_gd'] = compute_pi_predicted_gd(feats)
    feats['pi_gd_neu'] = compute_pi_predicted_gd(feats, neutral=feats['neutral'].values)
    key = ['date', 'home_team', 'away_team']
    pi_map = feats[key + ['pi_gd', 'pi_gd_neu']].drop_duplicates(key).copy()
    pi_map['date'] = pi_map['date'].dt.strftime('%Y-%m-%d')

    # 2) walkforward 評価試合にマージ
    preds = pd.read_csv(os.path.join(BASE_DIR, "data/processed/walkforward_predictions.csv"))
    preds['date'] = pd.to_datetime(preds['date']).dt.strftime('%Y-%m-%d')
    preds = preds.merge(pi_map, on=key, how='left')
    n_missing = preds['pi_gd'].isna().sum()
    preds = preds.dropna(subset=['pi_gd']).reset_index(drop=True)
    print(f"評価試合: {len(preds)} (pi_gd 欠損 {n_missing} 件は除外)")

    y = preds['actual'].map({'H': 0, 'D': 1, 'A': 2}).values

    # 3) pi の得失点差 → λペア。総得点スケールは現行アンサンブルに合わせる
    mu_tot = float((preds['lambda_home'] + preds['lambda_away']).mean())

    def to_lam(gd):
        return (np.clip((mu_tot + gd) / 2.0, 0.05, None),
                np.clip((mu_tot - gd) / 2.0, 0.05, None))
    lam_pi_h, lam_pi_a = to_lam(preds['pi_gd'].values)
    lam_pin_h, lam_pin_a = to_lam(preds['pi_gd_neu'].values)
    print(f"総得点スケール μ_tot={mu_tot:.3f}（現行λの平均総得点に一致させた）")

    # 4) λ構成（pi=素のpi-ratings, piN=中立地対応版）
    lam_p_h, lam_p_a = preds['lam_p_h'].values, preds['lam_p_a'].values
    lam_l_h, lam_l_a = preds['lam_l_h'].values, preds['lam_l_a'].values
    configs = {
        'P+L (現行)':  ((lam_p_h + lam_l_h) / 2, (lam_p_a + lam_l_a) / 2),
        'pi 単体':     (lam_pi_h, lam_pi_a),
        'P+L+pi':      ((lam_p_h + lam_l_h + lam_pi_h) / 3, (lam_p_a + lam_l_a + lam_pi_a) / 3),
        'piN 単体':    (lam_pin_h, lam_pin_a),
        'P+L+piN':     ((lam_p_h + lam_l_h + lam_pin_h) / 3, (lam_p_a + lam_l_a + lam_pin_a) / 3),
    }
    cls_lx = (preds[['p_home_cls', 'p_draw_cls', 'p_away_cls']].values
              + preds[['p_home_xcls', 'p_draw_xcls', 'p_away_xcls']].values) / 2.0

    wc = preds['is_wc'].values
    print(f"\n=== λ成分としての pi-ratings 評価 (ρ={RHO}, cls_blend={W_CLS}, n={len(preds)}) ===")
    print(f"{'構成':<14}{'LL(λのみ)':>11}{'LL(+分類器)':>12}{'Brier(+分)':>11}{'W杯LL(+分)':>12}")
    for name, (lh, la) in configs.items():
        mat = dc_outcome_probs(lh, la)
        blended = (1 - W_CLS) * mat + W_CLS * cls_lx
        ll_mat = logloss(mat, y)
        ll_bl = logloss(blended, y)
        br_bl = brier(blended, y)
        ll_wc = logloss(blended[wc], y[wc]) if wc.sum() else float('nan')
        print(f"{name:<14}{ll_mat:>11.5f}{ll_bl:>12.5f}{br_bl:>11.5f}{ll_wc:>12.5f}")

    # 5) piの寄与重みを最適化: λ = (1-w)*λ_PL + w*λ_pi を w で掃引
    pl_h, pl_a = (lam_p_h + lam_l_h) / 2, (lam_p_a + lam_l_a) / 2
    print(f"\n=== pi寄与重み w の掃引 [λ=(1-w)P+L + w·pi] ===")
    print(f"{'w':>5}{'全体LL':>10}{'全体Brier':>11}{'W杯LL':>10}")
    for w in [0.0, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5]:
        lh = (1 - w) * pl_h + w * lam_pi_h
        la = (1 - w) * pl_a + w * lam_pi_a
        mat = dc_outcome_probs(lh, la)
        bl = (1 - W_CLS) * mat + W_CLS * cls_lx
        print(f"{w:>5.2f}{logloss(bl, y):>10.5f}{brier(bl, y):>11.5f}"
              f"{(logloss(bl[wc], y[wc]) if wc.sum() else float('nan')):>10.5f}")

    # 6) 参考: 相関
    real_gd = (preds['home_score'] - preds['away_score']).values
    print(f"\n参考: pi_gd vs 実得失点差 相関 = {np.corrcoef(preds['pi_gd'].values, real_gd)[0,1]:+.3f}"
          f" / 現行λ差 vs 実得失点差 相関 = "
          f"{np.corrcoef(preds['lambda_home']-preds['lambda_away'], real_gd)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
