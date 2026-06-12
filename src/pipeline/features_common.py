#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features_common.py - 特徴量定義と own/opp 行構築の共通モジュール

学習(train_model)・バックテスト(backtest)・2026年予測(simulator_2026)の3経路は
かつてそれぞれ独自に own/opp マッピングを実装しており、コピペバグの温床だった
（2026-06に計11箇所の取り違えを修正した経緯がある）。
本モジュールが唯一の正であり、3経路すべてがここを参照する。

特徴量を追加する場合は FEATURE_COLS と（per-team統計なら）TEAM_STAT_COLS に
追記するだけでよい。
"""
import itertools
import numpy as np
import pandas as pd

# per-team の rolling/ewm 統計カラム（features.csv では home_/away_ プレフィックス付き、
# simulator の team_feats 辞書ではプレフィックスなしで保持される）
_STATS = ['goals', 'conceded', 'win_rate', 'goals_weighted', 'conceded_weighted']
TS_COLS = [f"{s}{o}_{m}{w}" for s, o, m, w in
           itertools.product(_STATS, ['', '_official'], ['roll', 'ewm'], [5, 10])]

# 特徴量キーの末尾 _own/_opp を除いた名前 → per-team ソースカラム名（命名例外のみ記載）
_KEY_TO_SOURCE = {'same_conf': 'same_confederation'}

# 学習・予測で使用する特徴量リスト（順序も含めてこの定義が正）
FEATURE_COLS = [
    'elo_diff',
    'squad_value_diff',
    'squad_value_missing_own', 'squad_value_missing_opp',
    'rest_days_own', 'rest_days_opp',
    'altitude_diff_own', 'altitude_diff_opp',
    'last_wcup_matches_own',
    'last_wcup_matches_opp',

    # 実質ホームアドバンテージ
    'same_conf_own', 'same_conf_opp',
    'is_host_own', 'is_host_opp',

    # 基本 rolling
    'goals_roll5_own', 'conceded_roll5_own', 'win_rate_roll5_own',
    'goals_roll10_own', 'conceded_roll10_own', 'win_rate_roll10_own',
    'goals_weighted_roll5_own', 'conceded_weighted_roll5_own',
    'goals_weighted_roll10_own', 'conceded_weighted_roll10_own',
    'goals_roll5_opp', 'conceded_roll5_opp', 'win_rate_roll5_opp',
    'goals_roll10_opp', 'conceded_roll10_opp', 'win_rate_roll10_opp',
    'goals_weighted_roll5_opp', 'conceded_weighted_roll5_opp',
    'goals_weighted_roll10_opp', 'conceded_weighted_roll10_opp',

    # 基本 ewm
    'goals_ewm5_own', 'conceded_ewm5_own', 'win_rate_ewm5_own',
    'goals_ewm10_own', 'conceded_ewm10_own', 'win_rate_ewm10_own',
    'goals_weighted_ewm5_own', 'conceded_weighted_ewm5_own',
    'goals_weighted_ewm10_own', 'conceded_weighted_ewm10_own',
    'goals_ewm5_opp', 'conceded_ewm5_opp', 'win_rate_ewm5_opp',
    'goals_ewm10_opp', 'conceded_ewm10_opp', 'win_rate_ewm10_opp',
    'goals_weighted_ewm5_opp', 'conceded_weighted_ewm5_opp',
    'goals_weighted_ewm10_opp', 'conceded_weighted_ewm10_opp',

    # 公式戦優先 rolling
    'goals_official_roll5_own', 'conceded_official_roll5_own', 'win_rate_official_roll5_own',
    'goals_official_roll10_own', 'conceded_official_roll10_own', 'win_rate_official_roll10_own',
    'goals_weighted_official_roll5_own', 'conceded_weighted_official_roll5_own',
    'goals_weighted_official_roll10_own', 'conceded_weighted_official_roll10_own',
    'goals_official_roll5_opp', 'conceded_official_roll5_opp', 'win_rate_official_roll5_opp',
    'goals_official_roll10_opp', 'conceded_official_roll10_opp', 'win_rate_official_roll10_opp',
    'goals_weighted_official_roll5_opp', 'conceded_weighted_official_roll5_opp',
    'goals_weighted_official_roll10_opp', 'conceded_weighted_official_roll10_opp',

    # 公式戦優先 ewm
    'goals_official_ewm5_own', 'conceded_official_ewm5_own', 'win_rate_official_ewm5_own',
    'goals_official_ewm10_own', 'conceded_official_ewm10_own', 'win_rate_official_ewm10_own',
    'goals_weighted_official_ewm5_own', 'conceded_weighted_official_ewm5_own',
    'goals_weighted_official_ewm10_own', 'conceded_weighted_official_ewm10_own',
    'goals_official_ewm5_opp', 'conceded_official_ewm5_opp', 'win_rate_official_ewm5_opp',
    'goals_official_ewm10_opp', 'conceded_official_ewm10_opp', 'win_rate_official_ewm10_opp',
    'goals_weighted_official_ewm5_opp', 'conceded_weighted_official_ewm5_opp',
    'goals_weighted_official_ewm10_opp', 'conceded_weighted_official_ewm10_opp',

    'was_home'
]

# _own/_opp を持つ特徴量キーの一覧（グローバル特徴量を除く）
_GLOBAL_KEYS = {'elo_diff', 'squad_value_diff', 'was_home'}
PAIR_KEYS = [k for k in FEATURE_COLS if k not in _GLOBAL_KEYS]


def source_col(key):
    """'goals_roll5_own' → ('goals_roll5', 'own') のように分解する"""
    assert key.endswith('_own') or key.endswith('_opp'), key
    base, side = key[:-4], key[-3:]
    return _KEY_TO_SOURCE.get(base, base), side


def build_views_from_match_df(df, was_home_series=None):
    """home_/away_ プレフィックス付きの試合DataFrameから、
    (ホーム視点, アウェイ視点) の特徴量DataFrameペアを構築する（ベクトル化）。

    was_home_series: ホーム視点の was_home (省略時は neutral 列から導出)
    """
    if was_home_series is None:
        was_home_series = (~df['neutral'].astype(bool)).astype(int)

    home_view, away_view = {}, {}
    home_view['elo_diff'] = df['elo_diff']
    away_view['elo_diff'] = -df['elo_diff']
    home_view['squad_value_diff'] = df['squad_value_diff']
    away_view['squad_value_diff'] = -df['squad_value_diff']
    for key in PAIR_KEYS:
        base, side = source_col(key)
        if side == 'own':
            home_view[key] = df[f'home_{base}']
            away_view[key] = df[f'away_{base}']
        else:
            home_view[key] = df[f'away_{base}']
            away_view[key] = df[f'home_{base}']
    home_view['was_home'] = was_home_series
    away_view['was_home'] = pd.Series(0, index=df.index)

    return (pd.DataFrame(home_view)[FEATURE_COLS].reset_index(drop=True),
            pd.DataFrame(away_view)[FEATURE_COLS].reset_index(drop=True))


def expand_df_to_rows(df_slice):
    """試合DataFrameを「ホーム視点行・アウェイ視点行」を交互に並べた学習用DataFrameに展開する。
    （行順: 試合1ホーム, 試合1アウェイ, 試合2ホーム, ... walkforward等がこの順序に依存）"""
    df = df_slice.reset_index(drop=True)
    home_view, away_view = build_views_from_match_df(df)

    home_wins = df['home_score'] > df['away_score']
    away_wins = df['home_score'] < df['away_score']
    home_class = pd.Series(1, index=df.index) + home_wins.astype(int) - away_wins.astype(int)

    home_view['score'] = df['home_score']
    home_view['result_class'] = home_class
    away_view['score'] = df['away_score']
    away_view['result_class'] = 2 - home_class

    n = len(df)
    home_view.index = range(0, 2 * n, 2)
    away_view.index = range(1, 2 * n, 2)
    return pd.concat([home_view, away_view]).sort_index().reset_index(drop=True)


# チームIDカテゴリカル列（LGBMの簡易チーム埋め込み用。FEATURE_COLSには含めない）
CAT_COLS = ['own_id', 'opp_id']


def interleaved_teams(df):
    """expand_df_to_rows の行順（ホーム視点・アウェイ視点の交互）に対応する
    (own チーム名, opp チーム名) の配列を返す"""
    n = len(df)
    own = np.empty(2 * n, dtype=object)
    opp = np.empty(2 * n, dtype=object)
    own[0::2] = df['home_team'].values
    opp[0::2] = df['away_team'].values
    own[1::2] = df['away_team'].values
    opp[1::2] = df['home_team'].values
    return own, opp


def add_team_ids(X_df, own_teams, opp_teams, categories):
    """特徴量DataFrameにチームIDカテゴリカル列を追加して返す"""
    X = X_df.copy()
    X['own_id'] = pd.Categorical(own_teams, categories=categories).codes
    X['opp_id'] = pd.Categorical(opp_teams, categories=categories).codes
    return X


def build_row_from_team_dicts(feat_own, feat_opp, *, elo_diff, squad_value_diff,
                              same_conf_own, same_conf_opp, is_host_own, is_host_opp,
                              rest_days_own, rest_days_opp,
                              altitude_diff_own, altitude_diff_opp, was_home):
    """per-team特徴量辞書（プレフィックスなしキー）のペアから1試合分の特徴量dictを構築する。
    simulator_2026 (make_match_features) 用。"""
    row = {
        'elo_diff': elo_diff,
        'squad_value_diff': squad_value_diff,
        'was_home': was_home,
        'same_conf_own': same_conf_own, 'same_conf_opp': same_conf_opp,
        'is_host_own': is_host_own, 'is_host_opp': is_host_opp,
        'rest_days_own': rest_days_own, 'rest_days_opp': rest_days_opp,
        'altitude_diff_own': altitude_diff_own, 'altitude_diff_opp': altitude_diff_opp,
    }
    for key in PAIR_KEYS:
        if key in row:
            continue
        base, side = source_col(key)
        src = feat_own if side == 'own' else feat_opp
        row[key] = src[base]
    return row
