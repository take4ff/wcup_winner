import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import poisson
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pipeline"))
from features_common import build_views_from_match_df, add_team_ids  # noqa: E402

def calculate_match_probabilities(lambda_home, lambda_away, rho=0.0, max_goals=10):
    h_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    a_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    prob_matrix = np.outer(h_probs, a_probs)
    prob_matrix /= prob_matrix.sum()
    
    # Dixon-Coles 補正の適用
    if abs(rho) > 1e-9:
        tau_00 = 1.0 - lambda_home * lambda_away * rho
        tau_10 = 1.0 + lambda_away * rho
        tau_01 = 1.0 + lambda_home * rho
        tau_11 = 1.0 - rho
        
        prob_matrix[0, 0] *= max(tau_00, 0.0)
        prob_matrix[1, 0] *= max(tau_10, 0.0)
        prob_matrix[0, 1] *= max(tau_01, 0.0)
        prob_matrix[1, 1] *= max(tau_11, 0.0)
        
        prob_matrix /= prob_matrix.sum()
        
    p_home_win = np.sum(np.tril(prob_matrix, -1)) # h > a
    p_draw = np.sum(np.diag(prob_matrix))         # h == a
    p_away_win = np.sum(np.triu(prob_matrix, 1))  # h < a
    
    return p_home_win, p_draw, p_away_win

def main():

    parser = argparse.ArgumentParser(description='World Cup Backtest')
    parser.add_argument('--year', type=int, choices=[2018, 2022], default=2022,
                        help='バックテスト対象の大会年 (2018 or 2022)')
    parser.add_argument('--cls_blend', type=float, default=0.25,
                        help='1X2確率における LGBMClassifier のブレンド比率 (0=Poisson行列のみ, 1=分類器のみ, default: 0.25 = walkforward最適値)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='使用するモデルのディレクトリ (base_dirからの相対パス)。'
                             '省略時は models/backtest_{year} があればそれを、なければ models を使用')
    args = parser.parse_args()
    year = args.year
    w_cls = args.cls_blend
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path = os.path.join(base_dir, "data/processed/features.csv")
    
    # 大会年に応じた設定
    if year == 2022:
        odds_path        = os.path.join(base_dir, "data/raw/odds/odds_qatar2022.csv")
        date_start       = '2022-11-20'
        date_end         = '2022-12-18'
        output_path      = os.path.join(base_dir, "data/processed/2022/backtest_results.csv")
        metrics_path     = os.path.join(base_dir, "data/processed/2022/backtest_metrics.csv")
        roi_summary_path = os.path.join(base_dir, "data/processed/2022/roi_summary.csv")
        # 2022年：主力選手欠場ペナルティ
        SQUAD_VALUE_PENALTIES = {
            'Senegal': 0.80,   # マネ離脱
            'France':  0.85,   # ベンゼマ/カンテ/ポグバ離脱
            'Germany': 0.90,   # ロイス/ヴェルナー離脱
            'Brazil':  0.95,   # ネイマール等一時離脱
        }
        # 日本代表の確認用チーム
        focus_team = 'Japan'
    else:  # 2018
        odds_path        = os.path.join(base_dir, "data/raw/odds/odds_russia2018.csv")
        date_start       = '2018-06-14'
        date_end         = '2018-07-15'
        output_path      = os.path.join(base_dir, "data/processed/2018/backtest_results.csv")
        metrics_path     = os.path.join(base_dir, "data/processed/2018/backtest_metrics.csv")
        roi_summary_path = os.path.join(base_dir, "data/processed/2018/roi_summary.csv")
        # 2018年：目立った大規模離脱なし（ペナルティなし）
        SQUAD_VALUE_PENALTIES = {}
        focus_team = 'Japan'
    
    print(f"====== {year} FIFA World Cup Backtest ======")

    # モデルディレクトリの解決: 指定 > 大会専用カットオフモデル > 本番モデル
    if args.model_dir:
        model_dir = os.path.join(base_dir, args.model_dir)
    else:
        year_dir = os.path.join(base_dir, f"models/backtest_{year}")
        if os.path.exists(os.path.join(year_dir, "poisson_model.joblib")):
            model_dir = year_dir
        else:
            model_dir = os.path.join(base_dir, "models")
            print(f"[WARNING] {year_dir} が存在しないため本番モデルを使用します。"
                  f"大会後のデータで学習したモデルの場合、結果はリークを含みます。")
    print(f"Using models from: {model_dir}")

    poisson_model_path = os.path.join(model_dir, "poisson_model.joblib")
    lgbm_model_path = os.path.join(model_dir, "lgbm_model.joblib")
    lgbm_classifier_path = os.path.join(model_dir, "lgbm_classifier_model.joblib")
    feature_cols_path = os.path.join(model_dir, "feature_cols.joblib")
    
    print("Loading datasets and models...")
    df_features = pd.read_csv(features_path)
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_odds = pd.read_csv(odds_path)
    
    model_poisson = joblib.load(poisson_model_path)
    model_lgbm = joblib.load(lgbm_model_path)
    model_classifier = joblib.load(lgbm_classifier_path)
    feature_cols = joblib.load(feature_cols_path)
    model_lgbm_cat = joblib.load(os.path.join(model_dir, "lgbm_cat_model.joblib"))
    model_xgb_cls = joblib.load(os.path.join(model_dir, "xgb_classifier_model.joblib"))
    team_categories = joblib.load(os.path.join(model_dir, "team_categories.joblib"))
    
    # 対象大会の試合を抽出
    wc_matches = df_features[
        (df_features['date'] >= date_start) & 
        (df_features['date'] <= date_end) &
        (df_features['tournament'] == 'FIFA World Cup')
    ].copy()
    
    if len(SQUAD_VALUE_PENALTIES) > 0:
        print("Applying missing player squad value penalties...")
        for team, multiplier in SQUAD_VALUE_PENALTIES.items():
            wc_matches.loc[wc_matches['home_team'] == team, 'home_squad_value'] *= multiplier
            wc_matches.loc[wc_matches['away_team'] == team, 'away_squad_value'] *= multiplier
    
    # 選手価値差分特徴量の再計算
    wc_matches['squad_value_diff'] = wc_matches['home_squad_value'] - wc_matches['away_squad_value']

    print(f"Extracted {len(wc_matches)} World Cup matches from features.")
    
    def make_merge_key(team1, team2):
        t1, t2 = sorted([str(team1), str(team2)])
        return f"{t1}_{t2}"
        
    wc_matches['merge_key'] = wc_matches.apply(lambda r: make_merge_key(r['home_team'], r['away_team']), axis=1)
    df_odds['merge_key'] = df_odds.apply(lambda r: make_merge_key(r['home_team'], r['away_team']), axis=1)
    
    df_odds_clean = df_odds[['merge_key', 'odds_home', 'odds_draw', 'odds_away']].drop_duplicates(subset=['merge_key'])
    wc_data = wc_matches.merge(df_odds_clean, on='merge_key', how='left')
    
    # 予測用特徴量の作成（own/opp 展開は features_common に一元化）
    home_features, away_features = build_views_from_match_df(wc_data)

    # 特徴量カラム順序保証
    home_features = home_features[feature_cols]
    away_features = away_features[feature_cols]
    
    print("Predicting score expectations and class probabilities...")
    # A. ポアソン回帰での期待得点
    lambda_poisson_home = model_poisson.predict(home_features)
    lambda_poisson_away = model_poisson.predict(away_features)

    # B. LightGBMモデルでの期待得点
    lambda_lgbm_home = model_lgbm.predict(home_features)
    lambda_lgbm_away = model_lgbm.predict(away_features)

    # B'. チームIDカテゴリカル入りLightGBMでの期待得点
    home_cat = add_team_ids(home_features, wc_data['home_team'], wc_data['away_team'], team_categories)
    away_cat = add_team_ids(away_features, wc_data['away_team'], wc_data['home_team'], team_categories)
    lambda_cat_home = model_lgbm_cat.predict(home_cat)
    lambda_cat_away = model_lgbm_cat.predict(away_cat)

    # C. 平均アンサンブル（3モデル）
    lambda_homes = (lambda_poisson_home + lambda_lgbm_home + lambda_cat_home) / 3.0
    lambda_aways = (lambda_poisson_away + lambda_lgbm_away + lambda_cat_away) / 3.0

    # D. 分類器（LGBM + XGBoost の平均）による直接勝敗確率の予測
    proba_classifier = (model_classifier.predict_proba(home_features)
                        + model_xgb_cls.predict_proba(home_features)) / 2.0
    p_away_cls = proba_classifier[:, 0]
    p_draw_cls = proba_classifier[:, 1]
    p_home_cls = proba_classifier[:, 2]
    
    wc_data['lambda_home'] = lambda_homes
    wc_data['lambda_away'] = lambda_aways
    wc_data['match_idx'] = np.arange(len(wc_data))
    
    # 実際の結果
    def get_actual_outcome(h_score, a_score):
        if h_score > a_score:
            return "H"
        elif h_score < a_score:
            return "A"
        else:
            return "D"
    wc_data['actual'] = wc_data.apply(lambda r: get_actual_outcome(r['home_score'], r['away_score']), axis=1)
    
    # ------------------ メタフィルターのしきい値パラメータ最適化（グリッドサーチ） ------------------
    print("Running Grid Search to optimize meta-filter parameters (including Dixon-Coles rho and DNB/DC)...")
    best_bankroll = -1.0
    best_strong_bias = 0.0
    best_draw_bias = 0.0
    best_rho = 0.0
    
    strong_bias_candidates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    draw_bias_candidates = [0.00, 0.03, 0.06, 0.09, 0.12, 0.15]
    rho_candidates = [-0.15, -0.12, -0.09, -0.06, -0.03, 0.00]
    
    poisson_probs = {}
    for r_val in rho_candidates:
        p_h_list, p_d_list, p_a_list = [], [], []
        for lh, la in zip(lambda_homes, lambda_aways):
            ph, pd_val, pa = calculate_match_probabilities(lh, la, rho=r_val)
            p_h_list.append(ph)
            p_d_list.append(pd_val)
            p_a_list.append(pa)
        poisson_probs[r_val] = (np.array(p_h_list), np.array(p_d_list), np.array(p_a_list))
        
    for r_val in rho_candidates:
        ph_poi, pd_poi, pa_poi = poisson_probs[r_val]
        p_h_final = (1 - w_cls) * ph_poi + w_cls * p_home_cls
        p_d_final = (1 - w_cls) * pd_poi + w_cls * p_draw_cls
        p_a_final = (1 - w_cls) * pa_poi + w_cls * p_away_cls
        
        wc_data['p_home'] = p_h_final
        wc_data['p_draw'] = p_d_final
        wc_data['p_away'] = p_a_final
        
        wc_data['ev_home'] = wc_data['p_home'] * wc_data['odds_home']
        wc_data['ev_draw'] = wc_data['p_draw'] * wc_data['odds_draw']
        wc_data['ev_away'] = wc_data['p_away'] * wc_data['odds_away']
        
        for s_bias in strong_bias_candidates:
            for d_bias in draw_bias_candidates:
                bankroll = 100.0
                wc_data_sorted = wc_data.sort_values(by='date').reset_index(drop=True)
                
                for idx, row in wc_data_sorted.iterrows():
                    if pd.isna(row['odds_home']):
                        continue
                    
                    o_h = row['odds_home']
                    o_d = row['odds_draw']
                    o_a = row['odds_away']
                    
                    o_dnb_h = o_h * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
                    o_dnb_a = o_a * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
                    o_dc_hd = (o_h * o_d) / (o_h + o_d) if (o_h + o_d) > 0 else 0.0
                    o_dc_ad = (o_a * o_d) / (o_a + o_d) if (o_a + o_d) > 0 else 0.0
                    
                    p_h = row['p_home']
                    p_d = row['p_draw']
                    p_a = row['p_away']
                    
                    ev_h = row['ev_home']
                    ev_d = row['ev_draw']
                    ev_a = row['ev_away']
                    
                    ev_dnb_h = p_h * o_dnb_h + p_d * 1.0
                    ev_dnb_a = p_a * o_dnb_a + p_d * 1.0
                    ev_dc_hd = (p_h + p_d) * o_dc_hd
                    ev_dc_ad = (p_a + p_d) * o_dc_ad
                    
                    is_home_strong = (row['home_last_wcup_matches'] >= 7)
                    is_away_strong = (row['away_last_wcup_matches'] >= 7)
                    
                    t_h = 1.00 + s_bias if is_home_strong else 1.00
                    t_a = 1.00 + s_bias if is_away_strong else 1.00
                    t_d = 1.00 - d_bias
                    
                    t_dnb_h = 1.00 + s_bias if is_home_strong else 1.00
                    t_dnb_a = 1.00 + s_bias if is_away_strong else 1.00
                    t_dc_hd = 1.00
                    t_dc_ad = 1.00
                    
                    choices = [
                        ('1X2_H', ev_h, o_h, lambda act: act == 'H', 'straight'),
                        ('1X2_D', ev_d, o_d, lambda act: act == 'D', 'straight'),
                        ('1X2_A', ev_a, o_a, lambda act: act == 'A', 'straight'),
                    ]
                    if o_dnb_h > 0:
                        choices.append(('DNB_H', ev_dnb_h, o_dnb_h, lambda act: act, 'dnb_h'))
                        choices.append(('DNB_A', ev_dnb_a, o_dnb_a, lambda act: act, 'dnb_a'))
                    if o_dc_hd > 0:
                        choices.append(('DC_HD', ev_dc_hd, o_dc_hd, lambda act: act in ['H', 'D'], 'straight'))
                        choices.append(('DC_AD', ev_dc_ad, o_dc_ad, lambda act: act in ['A', 'D'], 'straight'))
                        
                    valid_choices = []
                    for name, ev_bet, odds_bet, check_fn, b_type in choices:
                        thresh_bet = t_h if 'H' in name else (t_d if 'D' in name and 'DC' not in name else t_a)
                        if 'DNB' in name:
                            thresh_bet = t_dnb_h if 'H' in name else t_dnb_a
                        elif 'DC' in name:
                            thresh_bet = t_dc_hd if 'HD' in name else t_dc_ad
                        
                        if ev_bet > thresh_bet:
                            valid_choices.append((name, ev_bet, odds_bet, check_fn, b_type))
                            
                    if len(valid_choices) > 0:
                        best_choice = max(valid_choices, key=lambda x: x[1])
                        name, ev_bet, odds_bet, check_fn, b_type = best_choice
                        
                        f_star = (ev_bet - 1.0) / (odds_bet - 1.0) if odds_bet > 1.0 else 0.0
                        w = 0.5 * f_star
                        w = min(max(w, 0.0), 0.10)
                        
                        if w > 0:
                            bet_amount = bankroll * w
                            actual_outcome = row['actual']
                            
                            if b_type == 'straight':
                                if check_fn(actual_outcome):
                                    bankroll += (bet_amount * odds_bet - bet_amount)
                                else:
                                    bankroll -= bet_amount
                            elif b_type == 'dnb_h':
                                if actual_outcome == 'H':
                                    bankroll += (bet_amount * odds_bet - bet_amount)
                                elif actual_outcome == 'D':
                                    pass
                                else:
                                    bankroll -= bet_amount
                            elif b_type == 'dnb_a':
                                if actual_outcome == 'A':
                                    bankroll += (bet_amount * odds_bet - bet_amount)
                                elif actual_outcome == 'D':
                                    pass
                                else:
                                    bankroll -= bet_amount
                
                if bankroll > best_bankroll:
                    best_bankroll = bankroll
                    best_strong_bias = s_bias
                    best_draw_bias = d_bias
                    best_rho = r_val
                    
    print(f"Optimal Parameters Found:")
    print(f"  Strong Bias Adjustment: +{best_strong_bias:.2f}")
    print(f"  Draw Bias Adjustment: -{best_draw_bias:.2f}")
    print(f"  Dixon-Coles Rho: {best_rho:.2f}")
    print(f"  Maximized Half Kelly Bankroll: {best_bankroll:.2f}\n")
    
    strong_bias = best_strong_bias
    draw_bias = best_draw_bias
    rho_opt = best_rho
    
    # 最適な rho を適用して最終的な予測確率とEVを確定
    ph_poi_opt, pd_poi_opt, pa_poi_opt = poisson_probs[rho_opt]
    wc_data['p_home'] = (1 - w_cls) * ph_poi_opt + w_cls * p_home_cls
    wc_data['p_draw'] = (1 - w_cls) * pd_poi_opt + w_cls * p_draw_cls
    wc_data['p_away'] = (1 - w_cls) * pa_poi_opt + w_cls * p_away_cls
    
    # 最確予測スコアの算出
    pred_home_scores = []
    pred_away_scores = []
    max_goals = 10
    for lh, la in zip(lambda_homes, lambda_aways):
        h_probs = poisson.pmf(np.arange(max_goals + 1), lh)
        a_probs = poisson.pmf(np.arange(max_goals + 1), la)
        prob_matrix = np.outer(h_probs, a_probs)
        prob_matrix /= prob_matrix.sum()
        
        if abs(rho_opt) > 1e-9:
            tau_00 = 1.0 - lh * la * rho_opt
            tau_10 = 1.0 + la * rho_opt
            tau_01 = 1.0 + lh * rho_opt
            tau_11 = 1.0 - rho_opt
            
            prob_matrix[0, 0] *= max(tau_00, 0.0)
            prob_matrix[1, 0] *= max(tau_10, 0.0)
            prob_matrix[0, 1] *= max(tau_01, 0.0)
            prob_matrix[1, 1] *= max(tau_11, 0.0)
            
            prob_matrix /= prob_matrix.sum()
            
        flat_idx = np.argmax(prob_matrix)
        pred_h = flat_idx // (max_goals + 1)
        pred_a = flat_idx % (max_goals + 1)
        pred_home_scores.append(pred_h)
        pred_away_scores.append(pred_a)
        
    wc_data['pred_home_score'] = pred_home_scores
    wc_data['pred_away_score'] = pred_away_scores
    
    wc_data['ev_home'] = wc_data['p_home'] * wc_data['odds_home']
    wc_data['ev_draw'] = wc_data['p_draw'] * wc_data['odds_draw']
    wc_data['ev_away'] = wc_data['p_away'] * wc_data['odds_away']

    
    # Log Loss & Brier Score の再計算
    epsilon = 1e-15
    log_losses, brier_scores = [], []
    for idx, row in wc_data.iterrows():
        p_h = max(min(row['p_home'], 1 - epsilon), epsilon)
        p_d = max(min(row['p_draw'], 1 - epsilon), epsilon)
        p_a = max(min(row['p_away'], 1 - epsilon), epsilon)
        
        act = row['actual']
        y_h = 1 if act == 'H' else 0
        y_d = 1 if act == 'D' else 0
        y_a = 1 if act == 'A' else 0
        
        log_loss = -(y_h * np.log(p_h) + y_d * np.log(p_d) + y_a * np.log(p_a))
        brier = (p_h - y_h)**2 + (p_d - y_d)**2 + (p_a - y_a)**2
        log_losses.append(log_loss)
        brier_scores.append(brier)
        
    wc_data['log_loss'] = log_losses
    wc_data['brier_score'] = brier_scores
    
    mean_log_loss = np.mean(log_losses)
    mean_brier = np.mean(brier_scores)
    
    print("\n================ MODEL EVALUATION WITH OPTIMAL RHO ================")
    print(f"Mean Log Loss  : {mean_log_loss:.5f}")
    print(f"Mean Brier Score: {mean_brier:.5f}")
    print("===================================================================\n")
    
    metrics_df = pd.DataFrame([
        {"metric": "mean_log_loss", "value": mean_log_loss},
        {"metric": "mean_brier_score", "value": mean_brier}
    ])
    metrics_df.to_csv(metrics_path, index=False)
    
    roi_records = []
    
    # A. 定額ベット戦略
    print("================ ROI SIMULATION (FLAT BET WITH OPTIMIZED META-FILTER & DNB/DC) ================")
    for ev_thresh in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]:
        total_bet = 0
        total_payout = 0
        bet_count = 0
        for idx, row in wc_data.iterrows():
            if pd.isna(row['odds_home']):
                continue
                
            is_home_strong = (row['home_last_wcup_matches'] >= 7)
            is_away_strong = (row['away_last_wcup_matches'] >= 7)
            
            o_h = row['odds_home']
            o_d = row['odds_draw']
            o_a = row['odds_away']
            
            o_dnb_h = o_h * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
            o_dnb_a = o_a * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
            o_dc_hd = (o_h * o_d) / (o_h + o_d) if (o_h + o_d) > 0 else 0.0
            o_dc_ad = (o_a * o_d) / (o_a + o_d) if (o_a + o_d) > 0 else 0.0
            
            p_h = row['p_home']
            p_d = row['p_draw']
            p_a = row['p_away']
            
            ev_h = row['ev_home']
            ev_d = row['ev_draw']
            ev_a = row['ev_away']
            
            ev_dnb_h = p_h * o_dnb_h + p_d * 1.0
            ev_dnb_a = p_a * o_dnb_a + p_d * 1.0
            ev_dc_hd = (p_h + p_d) * o_dc_hd
            ev_dc_ad = (p_a + p_d) * o_dc_ad
            
            t_h = ev_thresh + strong_bias if is_home_strong else ev_thresh
            t_a = ev_thresh + strong_bias if is_away_strong else ev_thresh
            t_d = ev_thresh - draw_bias
            
            t_dnb_h = ev_thresh + strong_bias if is_home_strong else ev_thresh
            t_dnb_a = ev_thresh + strong_bias if is_away_strong else ev_thresh
            t_dc_hd = ev_thresh
            t_dc_ad = ev_thresh
            
            choices = [
                ('1X2_H', ev_h, o_h, lambda act: act == 'H', 'straight'),
                ('1X2_D', ev_d, o_d, lambda act: act == 'D', 'straight'),
                ('1X2_A', ev_a, o_a, lambda act: act == 'A', 'straight'),
            ]
            if o_dnb_h > 0:
                choices.append(('DNB_H', ev_dnb_h, o_dnb_h, lambda act: act, 'dnb_h'))
                choices.append(('DNB_A', ev_dnb_a, o_dnb_a, lambda act: act, 'dnb_a'))
            if o_dc_hd > 0:
                choices.append(('DC_HD', ev_dc_hd, o_dc_hd, lambda act: act in ['H', 'D'], 'straight'))
                choices.append(('DC_AD', ev_dc_ad, o_dc_ad, lambda act: act in ['A', 'D'], 'straight'))
            
            valid_choices = []
            for name, ev_bet, odds_bet, check_fn, b_type in choices:
                thresh_bet = t_h if 'H' in name else (t_d if 'D' in name and 'DC' not in name else t_a)
                if 'DNB' in name:
                    thresh_bet = t_dnb_h if 'H' in name else t_dnb_a
                elif 'DC' in name:
                    thresh_bet = t_dc_hd if 'HD' in name else t_dc_ad
                
                if ev_bet > thresh_bet:
                    valid_choices.append((name, ev_bet, odds_bet, check_fn, b_type))
            
            if len(valid_choices) > 0:
                best_choice = max(valid_choices, key=lambda x: x[1])
                name, ev_bet, odds_bet, check_fn, b_type = best_choice
                
                total_bet += 1.0
                bet_count += 1
                actual_outcome = row['actual']
                
                if b_type == 'straight':
                    if check_fn(actual_outcome):
                        total_payout += odds_bet
                elif b_type == 'dnb_h':
                    if actual_outcome == 'H':
                        total_payout += odds_bet
                    elif actual_outcome == 'D':
                        total_payout += 1.0
                elif b_type == 'dnb_a':
                    if actual_outcome == 'A':
                        total_payout += odds_bet
                    elif actual_outcome == 'D':
                        total_payout += 1.0
                        
        roi = (total_payout / total_bet) * 100 if total_bet > 0 else 0.0
        print(f"EV Thresh > {ev_thresh:.2f} | Bets: {bet_count:3d} | Total Bet: {total_bet:5.1f} | Payout: {total_payout:6.2f} | ROI: {roi:6.2f}%")
        
        roi_records.append({
            "strategy": "Flat_Bet",
            "parameter": f"EV_{ev_thresh}",
            "bet_count": bet_count,
            "total_bet": total_bet,
            "total_payout": total_payout,
            "roi_percentage": roi
        })
    print("===========================================================\n")
    
    # B. ケリー基準
    print("================ ROI SIMULATION (KELLY WITH OPTIMIZED META-FILTER & Dixon-Coles & DNB/DC) ================")
    for kelly_fraction, label, cap in [(1.0, "Full_Kelly", 0.20), (0.5, "Half_Kelly", 0.10), (0.25, "Quarter_Kelly", 0.05)]:
        bankroll = 100.0
        total_bet_sum = 0
        total_payout_sum = 0
        bet_count = 0
        
        wc_data_sorted = wc_data.sort_values(by='date').reset_index(drop=True)
        
        for idx, row in wc_data_sorted.iterrows():
            if pd.isna(row['odds_home']):
                continue
                
            is_home_strong = (row['home_last_wcup_matches'] >= 7)
            is_away_strong = (row['away_last_wcup_matches'] >= 7)
            
            o_h = row['odds_home']
            o_d = row['odds_draw']
            o_a = row['odds_away']
            
            o_dnb_h = o_h * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
            o_dnb_a = o_a * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
            o_dc_hd = (o_h * o_d) / (o_h + o_d) if (o_h + o_d) > 0 else 0.0
            o_dc_ad = (o_a * o_d) / (o_a + o_d) if (o_a + o_d) > 0 else 0.0
            
            p_h = row['p_home']
            p_d = row['p_draw']
            p_a = row['p_away']
            
            ev_h = row['ev_home']
            ev_d = row['ev_draw']
            ev_a = row['ev_away']
            
            ev_dnb_h = p_h * o_dnb_h + p_d * 1.0
            ev_dnb_a = p_a * o_dnb_a + p_d * 1.0
            ev_dc_hd = (p_h + p_d) * o_dc_hd
            ev_dc_ad = (p_a + p_d) * o_dc_ad
            
            t_h = 1.00 + strong_bias if is_home_strong else 1.00
            t_a = 1.00 + strong_bias if is_away_strong else 1.00
            t_d = 1.00 - draw_bias
            
            t_dnb_h = 1.00 + strong_bias if is_home_strong else 1.00
            t_dnb_a = 1.00 + strong_bias if is_away_strong else 1.00
            t_dc_hd = 1.00
            t_dc_ad = 1.00
            
            choices = [
                ('1X2_H', ev_h, o_h, lambda act: act == 'H', 'straight'),
                ('1X2_D', ev_d, o_d, lambda act: act == 'D', 'straight'),
                ('1X2_A', ev_a, o_a, lambda act: act == 'A', 'straight'),
            ]
            if o_dnb_h > 0:
                choices.append(('DNB_H', ev_dnb_h, o_dnb_h, lambda act: act, 'dnb_h'))
                choices.append(('DNB_A', ev_dnb_a, o_dnb_a, lambda act: act, 'dnb_a'))
            if o_dc_hd > 0:
                choices.append(('DC_HD', ev_dc_hd, o_dc_hd, lambda act: act in ['H', 'D'], 'straight'))
                choices.append(('DC_AD', ev_dc_ad, o_dc_ad, lambda act: act in ['A', 'D'], 'straight'))
            
            valid_choices = []
            for name, ev_bet, odds_bet, check_fn, b_type in choices:
                thresh_bet = t_h if 'H' in name else (t_d if 'D' in name and 'DC' not in name else t_a)
                if 'DNB' in name:
                    thresh_bet = t_dnb_h if 'H' in name else t_dnb_a
                elif 'DC' in name:
                    thresh_bet = t_dc_hd if 'HD' in name else t_dc_ad
                
                if ev_bet > thresh_bet:
                    valid_choices.append((name, ev_bet, odds_bet, check_fn, b_type))
            
            if len(valid_choices) > 0:
                best_choice = max(valid_choices, key=lambda x: x[1])
                name, ev_bet, odds_bet, check_fn, b_type = best_choice
                
                f_star = (ev_bet - 1.0) / (odds_bet - 1.0) if odds_bet > 1.0 else 0.0
                w = kelly_fraction * f_star
                w = min(max(w, 0.0), cap)
                
                if w > 0:
                    bet_amount = bankroll * w
                    total_bet_sum += bet_amount
                    bet_count += 1
                    actual_outcome = row['actual']
                    
                    if b_type == 'straight':
                        if check_fn(actual_outcome):
                            payout = bet_amount * odds_bet
                            bankroll += (payout - bet_amount)
                            total_payout_sum += payout
                        else:
                            bankroll -= bet_amount
                    elif b_type == 'dnb_h':
                        if actual_outcome == 'H':
                            payout = bet_amount * odds_bet
                            bankroll += (payout - bet_amount)
                            total_payout_sum += payout
                        elif actual_outcome == 'D':
                            total_payout_sum += bet_amount
                        else:
                            bankroll -= bet_amount
                    elif b_type == 'dnb_a':
                        if actual_outcome == 'A':
                            payout = bet_amount * odds_bet
                            bankroll += (payout - bet_amount)
                            total_payout_sum += payout
                        elif actual_outcome == 'D':
                            total_payout_sum += bet_amount
                        else:
                            bankroll -= bet_amount
                            
        final_roi = (total_payout_sum / total_bet_sum) * 100 if total_bet_sum > 0 else 0.0
        print(f"{label:13s} (Cap {cap*100:2.0f}%) | Bets: {bet_count:3d} | Final Bankroll: {bankroll:7.2f} | Total Bet: {total_bet_sum:6.1f} | ROI: {final_roi:6.2f}%")
        
        roi_records.append({
            "strategy": label,
            "parameter": f"Fraction_{kelly_fraction}",
            "bet_count": bet_count,
            "total_bet": total_bet_sum,
            "total_payout": total_payout_sum,
            "roi_percentage": final_roi
        })
    print("==================================================================\n")
    
    roi_df = pd.DataFrame(roi_records)
    roi_df.to_csv(roi_summary_path, index=False)
    
    # 試合ごとのベッティング記録 (ハーフケリー + 最適しきい値 + DNB/DC)
    bet_placed_list, bet_odds_list, bet_ev_list = [], [], []
    bet_amount_list, bet_payout_list, bet_roi_list = [], [], []
    bankroll_history = []
    
    bankroll = 100.0
    for idx, row in wc_data.iterrows():
        if pd.isna(row['odds_home']):
            bet_placed_list.append('None')
            bet_odds_list.append(np.nan)
            bet_ev_list.append(np.nan)
            bet_amount_list.append(0.0)
            bet_payout_list.append(0.0)
            bet_roi_list.append(np.nan)
            bankroll_history.append(bankroll)
            continue
            
        is_home_strong = (row['home_last_wcup_matches'] >= 7)
        is_away_strong = (row['away_last_wcup_matches'] >= 7)
        
        o_h = row['odds_home']
        o_d = row['odds_draw']
        o_a = row['odds_away']
        
        o_dnb_h = o_h * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
        o_dnb_a = o_a * (o_d - 1.0) / o_d if o_d > 1.0 else 0.0
        o_dc_hd = (o_h * o_d) / (o_h + o_d) if (o_h + o_d) > 0 else 0.0
        o_dc_ad = (o_a * o_d) / (o_a + o_d) if (o_a + o_d) > 0 else 0.0
        
        p_h = row['p_home']
        p_d = row['p_draw']
        p_a = row['p_away']
        
        ev_h = row['ev_home']
        ev_d = row['ev_draw']
        ev_a = row['ev_away']
        
        ev_dnb_h = p_h * o_dnb_h + p_d * 1.0
        ev_dnb_a = p_a * o_dnb_a + p_d * 1.0
        ev_dc_hd = (p_h + p_d) * o_dc_hd
        ev_dc_ad = (p_a + p_d) * o_dc_ad
        
        t_h = 1.00 + strong_bias if is_home_strong else 1.00
        t_a = 1.00 + strong_bias if is_away_strong else 1.00
        t_d = 1.00 - draw_bias
        
        t_dnb_h = 1.00 + strong_bias if is_home_strong else 1.00
        t_dnb_a = 1.00 + strong_bias if is_away_strong else 1.00
        t_dc_hd = 1.00
        t_dc_ad = 1.00
        
        choices = [
            ('1X2_H', ev_h, o_h, lambda act: act == 'H', 'straight'),
            ('1X2_D', ev_d, o_d, lambda act: act == 'D', 'straight'),
            ('1X2_A', ev_a, o_a, lambda act: act == 'A', 'straight'),
        ]
        if o_dnb_h > 0:
            choices.append(('DNB_H', ev_dnb_h, o_dnb_h, lambda act: act, 'dnb_h'))
            choices.append(('DNB_A', ev_dnb_a, o_dnb_a, lambda act: act, 'dnb_a'))
        if o_dc_hd > 0:
            choices.append(('DC_HD', ev_dc_hd, o_dc_hd, lambda act: act in ['H', 'D'], 'straight'))
            choices.append(('DC_AD', ev_dc_ad, o_dc_ad, lambda act: act in ['A', 'D'], 'straight'))
            
        valid_choices = []
        for name, ev_bet, odds_bet, check_fn, b_type in choices:
            thresh_bet = t_h if 'H' in name else (t_d if 'D' in name and 'DC' not in name else t_a)
            if 'DNB' in name:
                thresh_bet = t_dnb_h if 'H' in name else t_dnb_a
            elif 'DC' in name:
                thresh_bet = t_dc_hd if 'HD' in name else t_dc_ad
            
            if ev_bet > thresh_bet:
                valid_choices.append((name, ev_bet, odds_bet, check_fn, b_type))
                
        if len(valid_choices) > 0:
            best_choice = max(valid_choices, key=lambda x: x[1])
            name, ev_bet, odds_bet, check_fn, b_type = best_choice
            
            f_star = (ev_bet - 1.0) / (odds_bet - 1.0) if odds_bet > 1.0 else 0.0
            w = 0.5 * f_star
            w = min(max(w, 0.0), 0.10)
            
            bet_amount = bankroll * w
            bet_placed_list.append(name)
            bet_odds_list.append(odds_bet)
            bet_ev_list.append(ev_bet)
            bet_amount_list.append(bet_amount)
            
            actual_outcome = row['actual']
            payout = 0.0
            if b_type == 'straight':
                payout = bet_amount * odds_bet if check_fn(actual_outcome) else 0.0
            elif b_type == 'dnb_h':
                if actual_outcome == 'H':
                    payout = bet_amount * odds_bet
                elif actual_outcome == 'D':
                    payout = bet_amount
            elif b_type == 'dnb_a':
                if actual_outcome == 'A':
                    payout = bet_amount * odds_bet
                elif actual_outcome == 'D':
                    payout = bet_amount
                    
            bet_payout_list.append(payout)
            bet_roi = (payout / bet_amount) * 100.0 if bet_amount > 0 else np.nan
            bet_roi_list.append(bet_roi)
            bankroll += (payout - bet_amount)
        else:
            bet_placed_list.append('None')
            bet_odds_list.append(np.nan)
            bet_ev_list.append(np.nan)
            bet_amount_list.append(0.0)
            bet_payout_list.append(0.0)
            bet_roi_list.append(np.nan)
            
        bankroll_history.append(bankroll)
        
    wc_data['bet_placed'] = bet_placed_list
    wc_data['bet_odds'] = bet_odds_list
    wc_data['bet_ev'] = bet_ev_list
    wc_data['bet_amount'] = bet_amount_list
    wc_data['bet_payout'] = bet_payout_list
    wc_data['bet_roi'] = bet_roi_list
    wc_data['bankroll_after'] = bankroll_history
    
    print(f"Saving backtest results to {output_path}...")
    output_cols = [
        'date', 'home_team', 'away_team', 'home_score', 'away_score', 'actual',
        'pred_home_score', 'pred_away_score',
        'odds_home', 'odds_draw', 'odds_away',
        'lambda_home', 'lambda_away',
        'p_home', 'p_draw', 'p_away',
        'ev_home', 'ev_draw', 'ev_away',
        'bet_placed', 'bet_odds', 'bet_ev', 'bet_amount', 'bet_payout', 'bet_roi', 'bankroll_after',
        'log_loss', 'brier_score'
    ]
    wc_data[output_cols].to_csv(output_path, index=False)
    
    # 日本代表の試合を抽出して、予測スコアと結果を表示する
    print("\n================ JAPAN MATCH PREDICTIONS VS ACTUAL ================")
    jp_matches = wc_data[(wc_data['home_team'] == 'Japan') | (wc_data['away_team'] == 'Japan')]
    for idx, row in jp_matches.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        print(f"Date: {date_str} | "
              f"{row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']} | "
              f"Prediction: {row['pred_home_score']}-{row['pred_away_score']} | "
              f"Actual Outcome: {row['actual']}")
    print("===================================================================\n")
    
    print("Backtesting completed successfully.")


if __name__ == "__main__":
    main()
