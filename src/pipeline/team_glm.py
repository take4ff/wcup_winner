#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
team_glm.py - チーム攻撃力・守備力のPoisson GLM（Dixon-Coles型レーティングモデル）

各チームの得点期待値を
    log(λ) = 切片 + 攻撃力(自チーム) - 守備力(相手) + ホーム補正 × was_home
としてリッジ正則化Poisson回帰で推定する。特徴量ベースの既存モデルと異なり
「チームの固有値」を直接持つため、アンサンブルに異質な視点を加える。

- 学習データ: 全国際試合（デフォルト2000年以降）
- 時間減衰: 半減期 half_life 年の指数重み（レーティング追従のための減衰で、
  特徴量モデルで不採用としたサンプル減衰とは役割が異なる）
- 出場試合が少ないチームは 'OTHER' バケットに集約
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

OTHER = "__OTHER__"


class TeamGLM:
    def __init__(self, half_life=2.5, alpha=1e-3, min_matches=10, train_start='2000-01-01'):
        self.half_life = half_life
        self.alpha = alpha
        self.min_matches = min_matches
        self.train_start = train_start
        self.attack_ = {}
        self.defense_ = {}
        self.intercept_ = 0.0
        self.home_coef_ = 0.0

    def _team_index(self, teams_series):
        counts = teams_series.value_counts()
        teams = sorted(counts[counts >= self.min_matches].index)
        return {t: i for i, t in enumerate(teams + [OTHER])}

    def fit(self, results_df, train_end=None):
        df = results_df.dropna(subset=['home_score', 'away_score']).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= self.train_start]
        if train_end:
            df = df[df['date'] < train_end]

        idx = self._team_index(pd.concat([df['home_team'], df['away_team']]))
        n = len(idx)
        get = lambda t: idx.get(t, idx[OTHER])

        # 時間減衰重み（試合単位 → 2行）
        ref = pd.Timestamp(train_end) if train_end else df['date'].max()
        age = (ref - df['date']).dt.days / 365.25
        w_match = np.power(0.5, age / self.half_life).values
        weights = np.repeat(w_match, 2)

        # 行: 各試合×2（ホーム得点行・アウェイ得点行）
        # 列: [attack_0..n-1, defense_0..n-1, home_flag]
        m = len(df)
        rows, cols, vals = [], [], []
        y = np.empty(2 * m)
        home_idx = df['home_team'].map(get).values
        away_idx = df['away_team'].map(get).values
        not_neutral = (~df['neutral'].astype(bool)).astype(float).values

        for k in range(m):
            r = 2 * k
            rows += [r, r, r]
            cols += [home_idx[k], n + away_idx[k], 2 * n]
            vals += [1.0, -1.0, not_neutral[k]]
            y[r] = df['home_score'].iloc[k]

            r = 2 * k + 1
            rows += [r, r]
            cols += [away_idx[k], n + home_idx[k]]
            vals += [1.0, -1.0]
            y[r] = df['away_score'].iloc[k]

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 * n + 1))
        reg = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        try:
            reg.fit(X, y, sample_weight=weights)
        except (TypeError, ValueError):
            reg.fit(X.toarray(), y, sample_weight=weights)

        coef = reg.coef_
        inv = {i: t for t, i in idx.items()}
        self.attack_ = {inv[i]: coef[i] for i in range(n)}
        self.defense_ = {inv[i]: coef[n + i] for i in range(n)}
        self.home_coef_ = coef[2 * n]
        self.intercept_ = reg.intercept_
        print(f"TeamGLM fitted: {n - 1} teams (+OTHER), {m} matches, "
              f"home_coef={self.home_coef_:.3f}, half_life={self.half_life}y")
        return self

    def predict_lambda(self, team, opponent, was_home=0):
        att = self.attack_.get(team, self.attack_.get(OTHER, 0.0))
        dfn = self.defense_.get(opponent, self.defense_.get(OTHER, 0.0))
        eta = self.intercept_ + att - dfn + self.home_coef_ * was_home
        return float(np.exp(np.clip(eta, -3, 3)))

    def predict_lambdas(self, teams, opponents, was_home):
        return np.array([self.predict_lambda(t, o, h)
                         for t, o, h in zip(teams, opponents, was_home)])
