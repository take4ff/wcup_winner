"""
simulator_2026.py - 2026年FIFAワールドカップ（48チーム）モンテカルロ・シミュレーター

大会フォーマット:
  グループステージ: 12グループ × 4チーム... ではなく、正式には 12グループ × 4チーム
  ※ 2026年は当初3チーム/グループ案から 4チーム/12グループ に変更済み
  - 12グループ (A〜L)、各4チーム、総試合数 48
  - 各グループ上位2チーム (24チーム) + 8つのベスト3位 = 32チーム → R32進出
  - 以降: R32 → QF(8) → SF(4) → Final
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pipeline"))
from features_common import build_row_from_team_dicts, add_team_ids  # noqa: E402
from altitude import city_altitude, team_home_altitude, DEFAULT_ALTITUDE  # noqa: E402

# ===================================================================
# 2026年 FIFA World Cup グループ定義 (12グループ × 4チーム)
# ===================================================================
GROUPS_2026 = {
    # Group A: Mexico, South Africa, Korea Republic (→South Korea), Czech Republic
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    # Group B: Canada, Bosnia and Herzegovina, Qatar, Switzerland
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    # Group C: Brazil, Morocco, Haiti, Scotland
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    # Group D: United States, Paraguay, Australia, Turkey
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    # Group E: Germany, Curaçao, Ivory Coast, Ecuador
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    # Group F: Netherlands, Japan, Tunisia, Sweden (UEFAプレーオフB優勝)
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    # Group G: Belgium, Egypt, Iran, New Zealand
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    # Group H: Spain, Cape Verde, Saudi Arabia, Uruguay
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    # Group I: France, Senegal, Norway, Iraq (大陸間プレーオフ2優勝)
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    # Group J: Argentina, Algeria, Austria, Jordan
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    # Group K: Portugal, Uzbekistan, Colombia, DR Congo (大陸間プレーオフ1優勝)
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    # Group L: England, Croatia, Ghana, Panama
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

CONFEDERATIONS_2026 = {
    # UEFA
    'France': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Belgium': 'UEFA', 'Italy': 'UEFA',
    'Croatia': 'UEFA', 'Denmark': 'UEFA', 'Switzerland': 'UEFA', 'Serbia': 'UEFA',
    'Poland': 'UEFA', 'Romania': 'UEFA', 'Austria': 'UEFA', 'Slovakia': 'UEFA',
    'Czech Republic': 'UEFA', 'Scotland': 'UEFA', 'Turkey': 'UEFA', 'Georgia': 'UEFA',
    'Albania': 'UEFA', 'Slovenia': 'UEFA', 'Hungary': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Norway': 'UEFA',
    # CONMEBOL
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Bolivia': 'CONMEBOL', 'Venezuela': 'CONMEBOL',
    # AFC
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC', 'Uzbekistan': 'AFC', 'Jordan': 'AFC',
    # CAF
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Egypt': 'CAF', 'Nigeria': 'CAF',
    'Cameroon': 'CAF', 'Mali': 'CAF', 'South Africa': 'CAF', 'Ivory Coast': 'CAF',
    'Algeria': 'CAF', 'Tunisia': 'CAF', 'Ghana': 'CAF', 'Cape Verde': 'CAF',
    'DR Congo': 'CAF',
    'Haiti': 'CONCACAF',
    # CONCACAF
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Panama': 'CONCACAF', 'Honduras': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Jamaica': 'CONCACAF', 'Curaçao': 'CONCACAF',
    # AFC
    'Iraq': 'AFC',
    # UEFA extra
    'Sweden': 'UEFA',
    # OFC
    'New Zealand': 'OFC',
}

# 開催国
HOST_COUNTRIES = {'United States', 'Canada', 'Mexico'}

# ===================================================================
# FIFA公式 ノックアウトブラケット定義
# R32: (試合番号, スロットA, スロットB)
#   ('W', 'A') = グループA 1位 / ('R', 'A') = グループA 2位 /
#   ('T', 'ABCDF') = ベスト3位のうちグループA/B/C/D/Fのいずれか
# 出典: 2026 FIFA World Cup knockout stage (Matches 73-104)
# ===================================================================
R32_BRACKET = [
    (73, ('R', 'A'), ('R', 'B')),
    (74, ('W', 'E'), ('T', 'ABCDF')),
    (75, ('W', 'F'), ('R', 'C')),
    (76, ('W', 'C'), ('R', 'F')),
    (77, ('W', 'I'), ('T', 'CDFGH')),
    (78, ('R', 'E'), ('R', 'I')),
    (79, ('W', 'A'), ('T', 'CEFHI')),
    (80, ('W', 'L'), ('T', 'EHIJK')),
    (81, ('W', 'D'), ('T', 'BEFIJ')),
    (82, ('W', 'G'), ('T', 'AEHIJ')),
    (83, ('R', 'K'), ('R', 'L')),
    (84, ('W', 'H'), ('R', 'J')),
    (85, ('W', 'B'), ('T', 'EFGIJ')),
    (86, ('W', 'J'), ('R', 'H')),
    (87, ('W', 'K'), ('T', 'DEIJL')),
    (88, ('R', 'D'), ('R', 'G')),
]
# R16 (M89-96): R32の勝者同士の対戦 (試合番号ペア)
R16_PAIRS = [(74, 77), (73, 75), (76, 78), (79, 80), (83, 84), (81, 82), (86, 88), (85, 87)]
# QF (M97-100): R16勝者 (M89-96 を R16_PAIRS の並び順で 89..96 とする)
QF_PAIRS = [(89, 90), (93, 94), (91, 92), (95, 96)]
# SF (M101-102)
SF_PAIRS = [(97, 98), (99, 100)]


def assign_third_places(qualified_thirds):
    """ベスト3位8チームをR32の3位スロットに割り当てる。
    qualified_thirds: [(group, team), ...] ランキング順
    各スロットの許容グループ制約を満たす完全マッチングをバックトラッキングで探索する
    （FIFA Annex C の割当と厳密一致はしないが、自グループ回避等の制約は同一）。
    返り値: {match_no: team}
    """
    slots = [(no, allowed) for no, a, b in R32_BRACKET
             for kind, allowed in [b] if kind == 'T']
    # 候補が少ないスロットから割り当てて探索を効率化
    slots.sort(key=lambda s: sum(1 for g, _ in qualified_thirds if g in s[1]))

    assignment = {}

    def backtrack(i, used):
        if i == len(slots):
            return True
        match_no, allowed = slots[i]
        for grp, team in qualified_thirds:
            if grp in allowed and grp not in used:
                assignment[match_no] = team
                used.add(grp)
                if backtrack(i + 1, used):
                    return True
                used.discard(grp)
                del assignment[match_no]
        return False

    if not backtrack(0, set()):
        # 万一完全マッチングが見つからない場合は残りを順に埋める（理論上は発生しない）
        used_teams = set(assignment.values())
        remaining = [t for _, t in qualified_thirds if t not in used_teams]
        for no, allowed in slots:
            if no not in assignment and remaining:
                assignment[no] = remaining.pop(0)
    return assignment


def get_confederation(team):
    return CONFEDERATIONS_2026.get(team, 'Other')


# 大会中のグループステージは中4〜5日が標準。予測時の休養日数は定数で近似する
TOURNAMENT_REST_DAYS = 4.0


def load_squad_penalties():
    """主力離脱などによる市場価値ペナルティ (data/raw/squad/squad_penalties_2026.csv) を読み込む
    CSV形式: team,multiplier,reason (multiplier は 0.0〜1.0, 例: 0.85 = 15%減)"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    path = os.path.join(base_dir, "data/raw/squad/squad_penalties_2026.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    penalties = {}
    for _, row in df.iterrows():
        try:
            penalties[row['team']] = float(row['multiplier'])
        except (ValueError, TypeError):
            continue
    if penalties:
        print(f"  Applying squad value penalties: {penalties}")
    return penalties


def load_team_features(features_df):
    """2026年W杯開幕前の最新チーム特徴量を取得"""
    team_features = {}
    all_teams = []
    for teams in GROUPS_2026.values():
        all_teams.extend(teams)

    squad_penalties = load_squad_penalties()

    for team in all_teams:
        # 直近の試合（大会中の消化試合も含む）からチーム状態を取得する。
        # features.csv はスコア確定済みの試合のみ持つため、日付上限は不要
        team_matches = features_df[
            (features_df['home_team'] == team) | (features_df['away_team'] == team)
        ].sort_values(by='date', ascending=False)

        if len(team_matches) > 0:
            last = team_matches.iloc[0]
            is_home = last['home_team'] == team
            prefix = 'home_' if is_home else 'away_'

            elo = last[f'{prefix}elo_after']
            squad_value = last[f'{prefix}squad_value'] * squad_penalties.get(team, 1.0)

            def safe_get(col):
                return last[col] if col in last.index else 1.2

            team_features[team] = {
                'elo': elo,
                'squad_value': squad_value,
                'squad_value_missing': last.get(f'{prefix}squad_value_missing', 0),
                'last_wcup_matches': last[f'{prefix}last_wcup_matches'],
                'conf': get_confederation(team),
                'goals_roll5':   safe_get(f'{prefix}goals_roll5'),
                'conceded_roll5': safe_get(f'{prefix}conceded_roll5'),
                'win_rate_roll5': safe_get(f'{prefix}win_rate_roll5'),
                'goals_roll10':  safe_get(f'{prefix}goals_roll10'),
                'conceded_roll10': safe_get(f'{prefix}conceded_roll10'),
                'win_rate_roll10': safe_get(f'{prefix}win_rate_roll10'),
                'goals_weighted_roll5':  safe_get(f'{prefix}goals_weighted_roll5'),
                'conceded_weighted_roll5': safe_get(f'{prefix}conceded_weighted_roll5'),
                'goals_weighted_roll10': safe_get(f'{prefix}goals_weighted_roll10'),
                'conceded_weighted_roll10': safe_get(f'{prefix}conceded_weighted_roll10'),
                'goals_ewm5':   safe_get(f'{prefix}goals_ewm5'),
                'conceded_ewm5': safe_get(f'{prefix}conceded_ewm5'),
                'win_rate_ewm5': safe_get(f'{prefix}win_rate_ewm5'),
                'goals_ewm10':  safe_get(f'{prefix}goals_ewm10'),
                'conceded_ewm10': safe_get(f'{prefix}conceded_ewm10'),
                'win_rate_ewm10': safe_get(f'{prefix}win_rate_ewm10'),
                'goals_weighted_ewm5':  safe_get(f'{prefix}goals_weighted_ewm5'),
                'conceded_weighted_ewm5': safe_get(f'{prefix}conceded_weighted_ewm5'),
                'goals_weighted_ewm10': safe_get(f'{prefix}goals_weighted_ewm10'),
                'conceded_weighted_ewm10': safe_get(f'{prefix}conceded_weighted_ewm10'),
                'goals_official_roll5':  safe_get(f'{prefix}goals_official_roll5'),
                'conceded_official_roll5': safe_get(f'{prefix}conceded_official_roll5'),
                'win_rate_official_roll5': safe_get(f'{prefix}win_rate_official_roll5'),
                'goals_official_roll10': safe_get(f'{prefix}goals_official_roll10'),
                'conceded_official_roll10': safe_get(f'{prefix}conceded_official_roll10'),
                'win_rate_official_roll10': safe_get(f'{prefix}win_rate_official_roll10'),
                'goals_weighted_official_roll5': safe_get(f'{prefix}goals_weighted_official_roll5'),
                'conceded_weighted_official_roll5': safe_get(f'{prefix}conceded_weighted_official_roll5'),
                'goals_weighted_official_roll10': safe_get(f'{prefix}goals_weighted_official_roll10'),
                'conceded_weighted_official_roll10': safe_get(f'{prefix}conceded_weighted_official_roll10'),
                'goals_official_ewm5':  safe_get(f'{prefix}goals_official_ewm5'),
                'conceded_official_ewm5': safe_get(f'{prefix}conceded_official_ewm5'),
                'win_rate_official_ewm5': safe_get(f'{prefix}win_rate_official_ewm5'),
                'goals_official_ewm10': safe_get(f'{prefix}goals_official_ewm10'),
                'conceded_official_ewm10': safe_get(f'{prefix}conceded_official_ewm10'),
                'win_rate_official_ewm10': safe_get(f'{prefix}win_rate_official_ewm10'),
                'goals_weighted_official_ewm5': safe_get(f'{prefix}goals_weighted_official_ewm5'),
                'conceded_weighted_official_ewm5': safe_get(f'{prefix}conceded_weighted_official_ewm5'),
                'goals_weighted_official_ewm10': safe_get(f'{prefix}goals_weighted_official_ewm10'),
                'conceded_weighted_official_ewm10': safe_get(f'{prefix}conceded_weighted_official_ewm10'),
            }
        else:
            print(f"  [WARNING] No data found for {team}, using default values.")
            team_features[team] = {
                'elo': 1500.0, 'squad_value': 50.0, 'squad_value_missing': 1,
                'last_wcup_matches': 0,
                'conf': get_confederation(team),
                'goals_roll5': 1.2, 'conceded_roll5': 1.2, 'win_rate_roll5': 0.5,
                'goals_roll10': 1.2, 'conceded_roll10': 1.2, 'win_rate_roll10': 0.5,
                'goals_weighted_roll5': 1.2, 'conceded_weighted_roll5': 1.2,
                'goals_weighted_roll10': 1.2, 'conceded_weighted_roll10': 1.2,
                'goals_ewm5': 1.2, 'conceded_ewm5': 1.2, 'win_rate_ewm5': 0.5,
                'goals_ewm10': 1.2, 'conceded_ewm10': 1.2, 'win_rate_ewm10': 0.5,
                'goals_weighted_ewm5': 1.2, 'conceded_weighted_ewm5': 1.2,
                'goals_weighted_ewm10': 1.2, 'conceded_weighted_ewm10': 1.2,
                'goals_official_roll5': 1.2, 'conceded_official_roll5': 1.2, 'win_rate_official_roll5': 0.5,
                'goals_official_roll10': 1.2, 'conceded_official_roll10': 1.2, 'win_rate_official_roll10': 0.5,
                'goals_weighted_official_roll5': 1.2, 'conceded_weighted_official_roll5': 1.2,
                'goals_weighted_official_roll10': 1.2, 'conceded_weighted_official_roll10': 1.2,
                'goals_official_ewm5': 1.2, 'conceded_official_ewm5': 1.2, 'win_rate_official_ewm5': 0.5,
                'goals_official_ewm10': 1.2, 'conceded_official_ewm10': 1.2, 'win_rate_official_ewm10': 0.5,
                'goals_weighted_official_ewm5': 1.2, 'conceded_weighted_official_ewm5': 1.2,
                'goals_weighted_official_ewm10': 1.2, 'conceded_weighted_official_ewm10': 1.2,
            }

    return team_features


def make_match_features(team_a, team_b, team_feats, venue_altitude=DEFAULT_ALTITUDE,
                        rest_days_a=TOURNAMENT_REST_DAYS, rest_days_b=TOURNAMENT_REST_DAYS):
    feat_a = team_feats[team_a]
    feat_b = team_feats[team_b]

    return build_row_from_team_dicts(
        feat_a, feat_b,
        elo_diff=feat_a['elo'] - feat_b['elo'],
        squad_value_diff=feat_a['squad_value'] - feat_b['squad_value'],
        same_conf_own=1 if feat_a['conf'] == 'CONCACAF' else 0,
        same_conf_opp=1 if feat_b['conf'] == 'CONCACAF' else 0,
        is_host_own=1 if team_a in HOST_COUNTRIES else 0,
        is_host_opp=1 if team_b in HOST_COUNTRIES else 0,
        rest_days_own=rest_days_a,
        rest_days_opp=rest_days_b,
        altitude_diff_own=venue_altitude - team_home_altitude(team_a),
        altitude_diff_opp=venue_altitude - team_home_altitude(team_b),
        # 開催国（米加墨）は自国開催試合が neutral=FALSE になるため was_home=1 を適用
        was_home=1 if team_a in HOST_COUNTRIES else 0,
    )


def load_group_fixtures():
    """2026年W杯グループステージの実日程（開催都市つき）を results.csv から取得"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    df = pd.read_csv(os.path.join(base_dir, "data/raw/match/results.csv"))
    df['date'] = pd.to_datetime(df['date'])
    fx = df[(df['tournament'] == 'FIFA World Cup') &
            (df['date'] >= '2026-06-01') & (df['date'] <= '2026-06-27')]
    return fx[['date', 'home_team', 'away_team', 'city']], df


def group_fixture_rows(team_feats):
    """グループステージ実日程の (pair一覧, 特徴量行) を、実開催地の標高と
    実日程ベースの休養日数つきで構築。ノックアウトや汎用対戦カードは
    中立(平地)・標準休養のまま、実日程分だけ上書きに使う。"""
    pairs, rows = [], []
    try:
        fixtures, all_matches = load_group_fixtures()
    except Exception as e:
        print(f"  [WARNING] 日程データの読み込みに失敗、試合別上書きをスキップ: {e}")
        return pairs, rows

    def rest_days(team, match_date):
        prev = all_matches[
            ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
            (all_matches['date'] < match_date)
        ]['date'].max()
        if pd.isna(prev):
            return 30.0
        return float(min((match_date - prev).days, 30))

    for r in fixtures.itertuples():
        if r.home_team in team_feats and r.away_team in team_feats:
            va = city_altitude(r.city)
            rd_h = rest_days(r.home_team, r.date)
            rd_a = rest_days(r.away_team, r.date)
            pairs.append((r.home_team, r.away_team))
            rows.append(make_match_features(r.home_team, r.away_team, team_feats,
                                            venue_altitude=va, rest_days_a=rd_h, rest_days_b=rd_a))
            pairs.append((r.away_team, r.home_team))
            rows.append(make_match_features(r.away_team, r.home_team, team_feats,
                                            venue_altitude=va, rest_days_a=rd_a, rest_days_b=rd_h))
    return pairs, rows


def _ensemble_lambdas(rows, pairs, feature_cols, model_poisson, model_lgbm,
                      model_lgbm_cat=None, team_categories=None):
    """特徴量行から3モデル平均（cat未指定時は2モデル平均）のλを計算"""
    X = pd.DataFrame(rows)[feature_cols]
    lam = model_poisson.predict(X) + model_lgbm.predict(X)
    n_models = 2
    if model_lgbm_cat is not None and team_categories is not None:
        own = [p[0] for p in pairs]
        opp = [p[1] for p in pairs]
        X_cat = add_team_ids(X, own, opp, team_categories)
        lam = lam + model_lgbm_cat.predict(X_cat)
        n_models = 3
    return lam / n_models


def precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols,
                           model_lgbm_cat=None, team_categories=None):
    all_teams = list(team_feats.keys())
    pairs, rows = [], []
    for t1 in all_teams:
        for t2 in all_teams:
            if t1 != t2:
                pairs.append((t1, t2))
                rows.append(make_match_features(t1, t2, team_feats))

    lambdas = _ensemble_lambdas(rows, pairs, feature_cols, model_poisson, model_lgbm,
                                model_lgbm_cat, team_categories)

    lambda_cache = {}
    for (t1, t2), lam in zip(pairs, lambdas):
        lambda_cache[(t1, t2)] = lam

    # グループステージの実日程分は、実開催地の標高で再計算して上書き
    fx_pairs, fx_rows = group_fixture_rows(team_feats)
    if fx_pairs:
        lams_fx = _ensemble_lambdas(fx_rows, fx_pairs, feature_cols, model_poisson, model_lgbm,
                                    model_lgbm_cat, team_categories)
        for p, lam in zip(fx_pairs, lams_fx):
            lambda_cache[p] = lam
        print(f"  Applied venue-altitude overrides for {len(fx_pairs)//2} group fixtures")
    return lambda_cache


def predict_match(team_a, team_b, lambda_cache, rho=-0.09, max_goals=10):
    lambda_a = lambda_cache[(team_a, team_b)]
    lambda_b = lambda_cache[(team_b, team_a)]
    goals = np.arange(max_goals + 1)
    prob_matrix = np.outer(poisson.pmf(goals, lambda_a), poisson.pmf(goals, lambda_b))
    prob_matrix /= prob_matrix.sum()
    if abs(rho) > 1e-9:
        prob_matrix[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 0.0)
        prob_matrix[1, 0] *= max(1.0 + lambda_b * rho, 0.0)
        prob_matrix[0, 1] *= max(1.0 + lambda_a * rho, 0.0)
        prob_matrix[1, 1] *= max(1.0 - rho, 0.0)
        prob_matrix /= prob_matrix.sum()
    flat = prob_matrix.flatten()
    idx = np.random.choice(len(flat), p=flat)
    return idx // (max_goals + 1), idx % (max_goals + 1)


def simulate_group_stage(lambda_cache):
    """グループステージ: 12グループ × 4チーム、各組上位2チーム確定 + 全3位チームを返す"""
    group_top2 = {}   # grp -> [1位, 2位]
    all_third = []    # (grp, チーム名, pts, gd, gf)

    for grp, teams in GROUPS_2026.items():
        pts = defaultdict(int)
        gf  = defaultdict(int)
        ga  = defaultdict(int)
        h2h = {}  # (勝者, 敗者) -> True

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                s1, s2 = predict_match(t1, t2, lambda_cache)
                gf[t1] += s1; ga[t1] += s2
                gf[t2] += s2; ga[t2] += s1
                if s1 > s2:
                    pts[t1] += 3
                    h2h[(t1, t2)] = True
                elif s1 < s2:
                    pts[t2] += 3
                    h2h[(t2, t1)] = True
                else:
                    pts[t1] += 1; pts[t2] += 1

        standings = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t]),
            reverse=True
        )
        # 勝点・得失点差・総得点が完全同一の2チームは直接対決の勝者を上位に（FIFA規則準拠）
        for k in range(len(standings) - 1):
            a, b = standings[k], standings[k + 1]
            if (pts[a], gf[a] - ga[a], gf[a]) == (pts[b], gf[b] - ga[b], gf[b]) \
                    and h2h.get((b, a)):
                standings[k], standings[k + 1] = b, a
        group_top2[grp] = standings[:2]
        third = standings[2]
        all_third.append((grp, third, pts[third], gf[third] - ga[third], gf[third]))

    # ベスト3位 8チームを選出 (pts→GD→GF順)。グループ情報を保持して返す
    all_third_sorted = sorted(all_third, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [(t[0], t[1]) for t in all_third_sorted[:8]]

    return group_top2, best_thirds


def simulate_knockout_match(t1, t2, lambda_cache):
    """90分 → 延長(期待得点を1/3に縮小したPoisson) → PK(50/50) の順に決着をつける"""
    s1, s2 = predict_match(t1, t2, lambda_cache)
    if s1 > s2: return t1, t2
    elif s1 < s2: return t2, t1

    # 延長戦 (30分 ≒ λ × 1/3)
    e1 = np.random.poisson(lambda_cache[(t1, t2)] / 3.0)
    e2 = np.random.poisson(lambda_cache[(t2, t1)] / 3.0)
    if e1 > e2: return t1, t2
    elif e1 < e2: return t2, t1

    # PK戦
    return (t1, t2) if np.random.rand() < 0.5 else (t2, t1)


def simulate_tournament(lambda_cache):
    group_top2, best_thirds = simulate_group_stage(lambda_cache)
    progress = {}

    # グループ落ちチームを記録
    for grp, teams in GROUPS_2026.items():
        for t in teams:
            progress[t] = 'GroupStage'

    # ベスト3位を公式ブラケットの3位スロットへ割当
    third_assignment = assign_third_places(best_thirds)

    # R32 対戦カードを公式ブラケットで構築
    def resolve_slot(slot, match_no):
        kind, val = slot
        if kind == 'W':
            return group_top2[val][0]
        elif kind == 'R':
            return group_top2[val][1]
        else:  # 'T'
            return third_assignment[match_no]

    winners = {}  # match_no -> 勝者
    for match_no, slot_a, slot_b in R32_BRACKET:
        ta = resolve_slot(slot_a, match_no)
        tb = resolve_slot(slot_b, match_no)
        progress[ta] = 'R32'
        progress[tb] = 'R32'
        win, _ = simulate_knockout_match(ta, tb, lambda_cache)
        winners[match_no] = win

    # R32の勝者にR32到達マークを上書きしつつR16へ
    for match_no_r16, (ma, mb) in enumerate(R16_PAIRS, start=89):
        ta, tb = winners[ma], winners[mb]
        progress[ta] = 'R16'
        progress[tb] = 'R16'
        win, _ = simulate_knockout_match(ta, tb, lambda_cache)
        winners[match_no_r16] = win

    # QF
    for match_no_qf, (ma, mb) in enumerate(QF_PAIRS, start=97):
        ta, tb = winners[ma], winners[mb]
        progress[ta] = 'QF'
        progress[tb] = 'QF'
        win, _ = simulate_knockout_match(ta, tb, lambda_cache)
        winners[match_no_qf] = win

    # SF
    finalists, losers = [], []
    for match_no_sf, (ma, mb) in enumerate(SF_PAIRS, start=101):
        ta, tb = winners[ma], winners[mb]
        progress[ta] = 'SF'
        progress[tb] = 'SF'
        win, loss = simulate_knockout_match(ta, tb, lambda_cache)
        progress[win] = 'Final'
        finalists.append(win)
        losers.append(loss)

    # 3位決定戦 (M103)
    third, _ = simulate_knockout_match(losers[0], losers[1], lambda_cache)
    progress[third] = '3rd'

    # 決勝 (M104)
    champion, runner_up = simulate_knockout_match(finalists[0], finalists[1], lambda_cache)
    progress[champion] = 'Champion'
    progress[runner_up] = '2nd'

    return progress


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path        = os.path.join(base_dir, "data/processed/features.csv")
    poisson_model_path   = os.path.join(base_dir, "models/poisson_model.joblib")
    lgbm_model_path      = os.path.join(base_dir, "models/lgbm_model.joblib")
    feature_cols_path    = os.path.join(base_dir, "models/feature_cols.joblib")
    output_path          = os.path.join(base_dir, "data/processed/2026/simulation_results.csv")

    print("Loading models and features...")
    df_features = pd.read_csv(features_path)
    df_features['date'] = pd.to_datetime(df_features['date'])

    model_poisson  = joblib.load(poisson_model_path)
    model_lgbm     = joblib.load(lgbm_model_path)
    feature_cols   = joblib.load(feature_cols_path)
    base_models_dir = os.path.dirname(poisson_model_path)
    model_lgbm_cat = joblib.load(os.path.join(base_models_dir, "lgbm_cat_model.joblib"))
    team_categories = joblib.load(os.path.join(base_models_dir, "team_categories.joblib"))

    print("Caching team features just before the 2026 World Cup...")
    team_feats = load_team_features(df_features)
    print(f"  {len(team_feats)} teams loaded.")

    print("Precomputing expected goals (lambdas) for all possible matchups...")
    lambda_cache = precompute_all_lambdas(team_feats, model_poisson, model_lgbm, feature_cols,
                                          model_lgbm_cat, team_categories)

    num_simulations = 10000
    print(f"Running Monte Carlo Simulation ({num_simulations} iterations)...\n")

    stage_counts = defaultdict(lambda: defaultdict(int))

    for sim in range(num_simulations):
        if (sim + 1) % 2000 == 0:
            print(f"  Completed {sim + 1} simulations...")

        progress = simulate_tournament(lambda_cache)
        for team, stage in progress.items():
            stage_counts[team][stage] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF', 'R16', 'R32']:
                stage_counts[team]['Reached_R32'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF', 'R16']:
                stage_counts[team]['Reached_R16'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF', 'QF']:
                stage_counts[team]['Reached_QF'] += 1
            if stage in ['Champion', '2nd', '3rd', 'Final', 'SF']:
                stage_counts[team]['Reached_SF'] += 1
            if stage in ['Champion', '2nd', 'Final']:
                stage_counts[team]['Reached_Final'] += 1
            if stage == 'Champion':
                stage_counts[team]['Champion_Count'] += 1

    rows = []
    for team in team_feats.keys():
        elo_val = round(team_feats[team]['elo'], 1)
        c = stage_counts[team]
        grp = next((g for g, ts in GROUPS_2026.items() if team in ts), '?')
        rows.append({
            'team': team, 'group': grp, 'elo': elo_val,
            'R32_Prob':    round((c['Reached_R32']   / num_simulations) * 100, 2),
            'R16_Prob':    round((c['Reached_R16']   / num_simulations) * 100, 2),
            'QF_Prob':     round((c['Reached_QF']    / num_simulations) * 100, 2),
            'SF_Prob':     round((c['Reached_SF']    / num_simulations) * 100, 2),
            'Final_Prob':  round((c['Reached_Final'] / num_simulations) * 100, 2),
            'Winner_Prob': round((c['Champion_Count'] / num_simulations) * 100, 2),
        })

    df_results = pd.DataFrame(rows).sort_values(by='Winner_Prob', ascending=False).reset_index(drop=True)

    print("\n================ TOP 15 SIMULATION RESULTS (WINNER PROBABILITY) ================")
    print(df_results.head(15).to_string(index=False))
    print("=================================================================================\n")

    focus_teams = ['Japan', 'Brazil', 'Argentina', 'France', 'England', 'Germany',
                   'Spain', 'United States', 'Canada', 'Mexico', 'Morocco', 'Portugal']
    print("================ SPECIFIC TEAMS PROBABILITY ================")
    print(df_results[df_results['team'].isin(focus_teams)].to_string(index=False))
    print("============================================================\n")

    df_results.to_csv(output_path, index=False)
    print(f"Simulation results saved to {output_path}")


if __name__ == "__main__":
    main()
