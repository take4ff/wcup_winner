import os
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def expand_df_to_rows(df_slice):
    rows = []
    for idx, row in df_slice.iterrows():
        if row['home_score'] > row['away_score']:
            home_class = 2
            away_class = 0
        elif row['home_score'] == row['away_score']:
            home_class = 1
            away_class = 1
        else:
            home_class = 0
            away_class = 2

        # ホーム視点の行
        rows.append({
            'elo_diff': row['elo_diff'],
            'squad_value_diff': row['squad_value_diff'],
            'last_wcup_matches_own': row['home_last_wcup_matches'],
            'last_wcup_matches_opp': row['away_last_wcup_matches'],
            'same_conf_own': row['home_same_confederation'],
            'same_conf_opp': row['away_same_confederation'],
            'is_host_own': row['home_is_host'],
            'is_host_opp': row['away_is_host'],
            
            # 基本 rolling & ewm
            'goals_roll5_own': row['home_goals_roll5'],
            'conceded_roll5_own': row['home_conceded_roll5'],
            'win_rate_roll5_own': row['home_win_rate_roll5'],
            'goals_roll10_own': row['home_goals_roll10'],
            'conceded_roll10_own': row['home_conceded_roll10'],
            'win_rate_roll10_own': row['home_win_rate_roll10'],
            'goals_weighted_roll5_own': row['home_goals_weighted_roll5'],
            'conceded_weighted_roll5_own': row['home_conceded_weighted_roll5'],
            'goals_weighted_roll10_own': row['home_goals_weighted_roll10'],
            'conceded_weighted_roll10_own': row['home_conceded_weighted_roll10'],
            
            'goals_roll5_opp': row['away_goals_roll5'],
            'conceded_roll5_opp': row['away_conceded_roll5'],
            'win_rate_roll5_opp': row['away_win_rate_roll5'],
            'goals_roll10_opp': row['away_goals_roll10'],
            'conceded_roll10_opp': row['away_conceded_roll10'],
            'win_rate_roll10_opp': row['away_win_rate_roll10'],
            'goals_weighted_roll5_opp': row['away_goals_weighted_roll5'],
            'conceded_weighted_roll5_opp': row['away_conceded_weighted_roll5'],
            'goals_weighted_roll10_opp': row['away_goals_weighted_roll10'],
            'conceded_weighted_roll10_opp': row['away_conceded_weighted_roll10'],
            
            'goals_ewm5_own': row['home_goals_ewm5'],
            'conceded_ewm5_own': row['home_conceded_ewm5'],
            'win_rate_ewm5_own': row['home_win_rate_ewm5'],
            'goals_ewm10_own': row['home_goals_ewm10'],
            'conceded_ewm10_own': row['home_conceded_ewm10'],
            'win_rate_ewm10_own': row['home_win_rate_ewm10'],
            'goals_weighted_ewm5_own': row['home_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_own': row['home_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_own': row['home_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_own': row['home_conceded_weighted_ewm10'],
            
            'goals_ewm5_opp': row['away_goals_ewm5'],
            'conceded_ewm5_opp': row['away_conceded_ewm5'],
            'win_rate_ewm5_opp': row['away_win_rate_ewm5'],
            'goals_ewm10_opp': row['away_goals_ewm10'],
            'conceded_ewm10_opp': row['away_conceded_ewm10'],
            'win_rate_ewm10_opp': row['away_win_rate_ewm10'],
            'goals_weighted_ewm5_opp': row['away_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_opp': row['away_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_opp': row['away_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_opp': row['away_conceded_weighted_ewm10'],

            # 公式戦優先
            'goals_official_roll5_own': row['home_goals_official_roll5'],
            'conceded_official_roll5_own': row['home_conceded_official_roll5'],
            'win_rate_official_roll5_own': row['home_win_rate_official_roll5'],
            'goals_official_roll10_own': row['home_goals_official_roll10'],
            'conceded_official_roll10_own': row['home_conceded_official_roll10'],
            'win_rate_official_roll10_own': row['home_win_rate_official_roll10'],
            'goals_weighted_official_roll5_own': row['home_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_own': row['home_conceded_weighted_official_roll5'],
            'goals_weighted_official_roll10_own': row['home_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_own': row['home_conceded_weighted_official_roll10'],
            'goals_official_roll5_opp': row['away_goals_official_roll5'],
            'conceded_official_roll5_opp': row['away_conceded_official_roll5'],
            'win_rate_official_roll5_opp': row['away_win_rate_official_roll5'],
            'goals_official_roll10_opp': row['away_goals_official_roll10'],
            'conceded_official_roll10_opp': row['away_conceded_official_roll10'],
            'win_rate_official_roll10_opp': row['away_win_rate_official_roll10'],
            'goals_weighted_official_roll5_opp': row['away_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_opp': row['away_conceded_official_roll5'],
            'goals_weighted_official_roll10_opp': row['away_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_opp': row['away_conceded_weighted_official_roll10'],

            'goals_official_ewm5_own': row['home_goals_official_ewm5'],
            'conceded_official_ewm5_own': row['home_conceded_official_ewm5'],
            'win_rate_official_ewm5_own': row['home_win_rate_official_ewm5'],
            'goals_official_ewm10_own': row['home_goals_official_ewm10'],
            'conceded_official_ewm10_own': row['home_conceded_official_ewm10'],
            'win_rate_official_ewm10_own': row['home_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_own': row['home_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_own': row['home_conceded_official_ewm5'],
            'goals_weighted_official_ewm10_own': row['home_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_own': row['home_conceded_weighted_official_ewm10'],
            'goals_official_ewm5_opp': row['away_goals_official_ewm5'],
            'conceded_official_ewm5_opp': row['away_conceded_official_ewm5'],
            'win_rate_official_ewm5_opp': row['away_win_rate_official_ewm5'],
            'goals_official_ewm10_opp': row['away_goals_official_ewm10'],
            'conceded_official_ewm10_opp': row['away_conceded_official_ewm10'],
            'win_rate_official_ewm10_opp': row['away_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_opp': row['away_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_opp': row['away_conceded_weighted_official_ewm5'],
            'goals_weighted_official_ewm10_opp': row['away_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_opp': row['away_conceded_weighted_official_ewm10'],
            
            'was_home': 1 if not row['neutral'] else 0,
            'score': row['home_score'],
            'result_class': home_class
        })
        # アウェイ視点
        rows.append({
            'elo_diff': -row['elo_diff'],
            'squad_value_diff': -row['squad_value_diff'],
            'last_wcup_matches_own': row['away_last_wcup_matches'],
            'last_wcup_matches_opp': row['home_last_wcup_matches'],
            'same_conf_own': row['away_same_confederation'],
            'same_conf_opp': row['home_same_confederation'],
            'is_host_own': row['away_is_host'],
            'is_host_opp': row['home_is_host'],
            
            # 基本 rolling & ewm
            'goals_roll5_own': row['away_goals_roll5'],
            'conceded_roll5_own': row['away_conceded_roll5'],
            'win_rate_roll5_own': row['away_win_rate_roll5'],
            'goals_roll10_own': row['away_goals_roll10'],
            'conceded_roll10_own': row['away_conceded_roll10'],
            'win_rate_roll10_own': row['away_win_rate_roll10'],
            'goals_weighted_roll5_own': row['away_goals_weighted_roll5'],
            'conceded_weighted_roll5_own': row['away_conceded_weighted_roll5'],
            'goals_weighted_roll10_own': row['away_goals_weighted_roll10'],
            'conceded_weighted_roll10_own': row['away_conceded_weighted_roll10'],
            'goals_roll5_opp': row['home_goals_roll5'],
            'conceded_roll5_opp': row['home_conceded_roll5'],
            'win_rate_roll5_opp': row['home_win_rate_roll5'],
            'goals_roll10_opp': row['home_goals_roll10'],
            'conceded_roll10_opp': row['home_conceded_roll10'],
            'win_rate_roll10_opp': row['home_win_rate_roll10'],
            'goals_weighted_roll5_opp': row['home_goals_weighted_roll5'],
            'conceded_weighted_roll5_opp': row['home_conceded_weighted_roll5'],
            'goals_weighted_roll10_opp': row['home_goals_weighted_roll10'],
            'conceded_weighted_roll10_opp': row['home_conceded_weighted_roll10'],
            
            'goals_ewm5_own': row['away_goals_ewm5'],
            'conceded_ewm5_own': row['away_conceded_ewm5'],
            'win_rate_ewm5_own': row['away_win_rate_ewm5'],
            'goals_ewm10_own': row['away_goals_ewm10'],
            'conceded_ewm10_own': row['away_conceded_ewm10'],
            'win_rate_ewm10_own': row['away_win_rate_ewm10'],
            'goals_weighted_ewm5_own': row['away_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_own': row['away_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_own': row['away_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_own': row['away_conceded_weighted_ewm10'],
            'goals_ewm5_opp': row['home_goals_ewm5'],
            'conceded_ewm5_opp': row['home_conceded_ewm5'],
            'win_rate_ewm5_opp': row['home_win_rate_ewm5'],
            'goals_ewm10_opp': row['home_goals_ewm10'],
            'conceded_ewm10_opp': row['home_conceded_ewm10'],
            'win_rate_ewm10_opp': row['home_win_rate_ewm10'],
            'goals_weighted_ewm5_opp': row['home_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_opp': row['home_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_opp': row['home_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_opp': row['home_conceded_weighted_ewm10'],

            # 公式戦優先
            'goals_official_roll5_own': row['away_goals_official_roll5'],
            'conceded_official_roll5_own': row['away_conceded_official_roll5'],
            'win_rate_official_roll5_own': row['away_win_rate_official_roll5'],
            'goals_official_roll10_own': row['away_goals_official_roll10'],
            'conceded_official_roll10_own': row['away_conceded_official_roll10'],
            'win_rate_official_roll10_own': row['away_win_rate_official_roll10'],
            'goals_weighted_official_roll5_own': row['away_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_own': row['away_conceded_official_roll5'],
            'goals_weighted_official_roll10_own': row['away_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_own': row['away_conceded_weighted_official_roll10'],
            'goals_official_roll5_opp': row['home_goals_official_roll5'],
            'conceded_official_roll5_opp': row['home_conceded_official_roll5'],
            'win_rate_official_roll5_opp': row['home_win_rate_official_roll5'],
            'goals_official_roll10_opp': row['home_goals_official_roll10'],
            'conceded_official_roll10_opp': row['home_conceded_official_roll10'],
            'win_rate_official_roll10_opp': row['home_win_rate_official_roll10'],
            'goals_weighted_official_roll5_opp': row['home_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_opp': row['home_conceded_official_roll5'],
            'goals_weighted_official_roll10_opp': row['home_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_opp': row['home_conceded_weighted_official_roll10'],

            'goals_official_ewm5_own': row['away_goals_official_ewm5'],
            'conceded_official_ewm5_own': row['away_conceded_official_ewm5'],
            'win_rate_official_ewm5_own': row['away_win_rate_official_ewm5'],
            'goals_official_ewm10_own': row['away_goals_official_ewm10'],
            'conceded_official_ewm10_own': row['away_conceded_official_ewm10'],
            'win_rate_official_ewm10_own': row['away_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_own': row['away_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_own': row['away_conceded_official_ewm5'],
            'goals_weighted_official_ewm10_own': row['away_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_own': row['away_conceded_weighted_official_ewm10'],
            'goals_official_ewm5_opp': row['home_goals_official_ewm5'],
            'conceded_official_ewm5_opp': row['home_conceded_official_ewm5'],
            'win_rate_official_ewm5_opp': row['home_win_rate_official_ewm5'],
            'goals_official_ewm10_opp': row['home_goals_official_ewm10'],
            'conceded_official_ewm10_opp': row['home_conceded_official_ewm10'],
            'win_rate_official_ewm10_opp': row['home_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_opp': row['home_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_opp': row['home_conceded_weighted_official_ewm5'],
            'goals_weighted_official_ewm10_opp': row['home_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_opp': row['home_conceded_weighted_official_ewm10'],
            
            'was_home': 0,
            'score': row['away_score'],
            'result_class': away_class
        })
    return pd.DataFrame(rows)

def optimize_lgbm_regressor(train_df, feature_cols, n_trials=30):
    print("Running Optuna for LGBMRegressor (Poisson)...")
    def objective(trial):
        params = {
            'objective': 'poisson',
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 10, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbosity': -1
        }
        tscv = TimeSeriesSplit(n_splits=5)
        losses = []
        for train_idx, val_idx in tscv.split(train_df):
            df_train = train_df.iloc[train_idx]
            df_val = train_df.iloc[val_idx]
            
            expanded_train = expand_df_to_rows(df_train)
            expanded_val = expand_df_to_rows(df_val)
            
            X_tr, y_tr = expanded_train[feature_cols], expanded_train['score']
            X_va, y_va = expanded_val[feature_cols], expanded_val['score']
            
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_va)
            preds = np.clip(preds, 1e-10, None)
            loss = np.mean(preds - y_va * np.log(preds))
            losses.append(loss)
            
        return np.mean(losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LGBMRegressor parameters: {study.best_params}")
    return study.best_params

def optimize_lgbm_classifier(train_df, feature_cols, n_trials=30):
    print("Running Optuna for LGBMClassifier (Multiclass)...")
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 10, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbosity': -1
        }
        tscv = TimeSeriesSplit(n_splits=5)
        losses = []
        for train_idx, val_idx in tscv.split(train_df):
            df_train = train_df.iloc[train_idx]
            df_val = train_df.iloc[val_idx]
            
            expanded_train = expand_df_to_rows(df_train)
            expanded_val = expand_df_to_rows(df_val)
            
            X_tr, y_tr_cls = expanded_train[feature_cols], expanded_train['result_class']
            X_va, y_va_cls = expanded_val[feature_cols], expanded_val['result_class']
            
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr_cls)
            
            proba = model.predict_proba(X_va)
            epsilon = 1e-15
            proba = np.clip(proba, epsilon, 1 - epsilon)
            
            one_hot = np.eye(3)[y_va_cls]
            loss = -np.mean(np.sum(one_hot * np.log(proba), axis=1))
            losses.append(loss)
            
        return np.mean(losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LGBMClassifier parameters: {study.best_params}")
    return study.best_params

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_path = os.path.join(base_dir, "data/processed/features.csv")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Reading features from {input_path}...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 学習データの期間設定 (2015-01-01 から 2022-11-19 まで)
    train_df = df[(df['date'] >= '2015-01-01') & (df['date'] < '2022-11-20')].copy()
    print(f"Number of training matches: {len(train_df)}")
    
    # データをチーム視点（対称）に展開して学習データを構築
    rows = []
    for idx, row in train_df.iterrows():
        # 勝敗クラスラベル判定 (2:勝, 1:分, 0:負)
        if row['home_score'] > row['away_score']:
            home_class = 2
            away_class = 0
        elif row['home_score'] == row['away_score']:
            home_class = 1
            away_class = 1
        else:
            home_class = 0
            away_class = 2

        # ホーム視点の行
        rows.append({
            'elo_diff': row['elo_diff'],
            'squad_value_diff': row['squad_value_diff'],
            'last_wcup_matches_own': row['home_last_wcup_matches'],
            'last_wcup_matches_opp': row['away_last_wcup_matches'],
            
            # 実質ホームアドバンテージ
            'same_conf_own': row['home_same_confederation'],
            'same_conf_opp': row['away_same_confederation'],
            'is_host_own': row['home_is_host'],
            'is_host_opp': row['away_is_host'],
            
            # 基本 rolling
            'goals_roll5_own': row['home_goals_roll5'],
            'conceded_roll5_own': row['home_conceded_roll5'],
            'win_rate_roll5_own': row['home_win_rate_roll5'],
            'goals_roll10_own': row['home_goals_roll10'],
            'conceded_roll10_own': row['home_conceded_roll10'],
            'win_rate_roll10_own': row['home_win_rate_roll10'],
            'goals_weighted_roll5_own': row['home_goals_weighted_roll5'],
            'conceded_weighted_roll5_own': row['home_conceded_weighted_roll5'],
            'goals_weighted_roll10_own': row['home_goals_weighted_roll10'],
            'conceded_weighted_roll10_own': row['home_conceded_weighted_roll10'],
            
            'goals_roll5_opp': row['away_goals_roll5'],
            'conceded_roll5_opp': row['away_conceded_roll5'],
            'win_rate_roll5_opp': row['away_win_rate_roll5'],
            'goals_roll10_opp': row['away_goals_roll10'],
            'conceded_roll10_opp': row['away_conceded_roll10'],
            'win_rate_roll10_opp': row['away_win_rate_roll10'],
            'goals_weighted_roll5_opp': row['away_goals_weighted_roll5'],
            'conceded_weighted_roll5_opp': row['away_conceded_weighted_roll5'],
            'goals_weighted_roll10_opp': row['away_goals_weighted_roll10'],
            'conceded_weighted_roll10_opp': row['away_conceded_weighted_roll10'],

            # 基本 ewm
            'goals_ewm5_own': row['home_goals_ewm5'],
            'conceded_ewm5_own': row['home_conceded_ewm5'],
            'win_rate_ewm5_own': row['home_win_rate_ewm5'],
            'goals_ewm10_own': row['home_goals_ewm10'],
            'conceded_ewm10_own': row['home_conceded_ewm10'],
            'win_rate_ewm10_own': row['home_win_rate_ewm10'],
            'goals_weighted_ewm5_own': row['home_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_own': row['home_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_own': row['home_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_own': row['home_conceded_weighted_ewm10'],
            
            'goals_ewm5_opp': row['away_goals_ewm5'],
            'conceded_ewm5_opp': row['away_conceded_ewm5'],
            'win_rate_ewm5_opp': row['away_win_rate_ewm5'],
            'goals_ewm10_opp': row['away_goals_ewm10'],
            'conceded_ewm10_opp': row['away_conceded_ewm10'],
            'win_rate_ewm10_opp': row['away_win_rate_ewm10'],
            'goals_weighted_ewm5_opp': row['away_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_opp': row['away_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_opp': row['away_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_opp': row['away_conceded_weighted_ewm10'],
            
            # 公式戦優先 rolling
            'goals_official_roll5_own': row['home_goals_official_roll5'],
            'conceded_official_roll5_own': row['home_conceded_official_roll5'],
            'win_rate_official_roll5_own': row['home_win_rate_official_roll5'],
            'goals_official_roll10_own': row['home_goals_official_roll10'],
            'conceded_official_roll10_own': row['home_conceded_official_roll10'],
            'win_rate_official_roll10_own': row['home_win_rate_official_roll10'],
            'goals_weighted_official_roll5_own': row['home_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_own': row['home_conceded_weighted_official_roll5'],
            'goals_weighted_official_roll10_own': row['home_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_own': row['home_conceded_weighted_official_roll10'],
            
            'goals_official_roll5_opp': row['away_goals_official_roll5'],
            'conceded_official_roll5_opp': row['away_conceded_official_roll5'],
            'win_rate_official_roll5_opp': row['away_win_rate_official_roll5'],
            'goals_official_roll10_opp': row['away_goals_official_roll10'],
            'conceded_official_roll10_opp': row['away_conceded_official_roll10'],
            'win_rate_official_roll10_opp': row['away_win_rate_official_roll10'],
            'goals_weighted_official_roll5_opp': row['away_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_opp': row['away_conceded_weighted_official_roll5'],
            'goals_weighted_official_roll10_opp': row['away_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_opp': row['away_conceded_weighted_official_roll10'],

            # 公式戦優先 ewm
            'goals_official_ewm5_own': row['home_goals_official_ewm5'],
            'conceded_official_ewm5_own': row['home_conceded_official_ewm5'],
            'win_rate_official_ewm5_own': row['home_win_rate_official_ewm5'],
            'goals_official_ewm10_own': row['home_goals_official_ewm10'],
            'conceded_official_ewm10_own': row['home_conceded_official_ewm10'],
            'win_rate_official_ewm10_own': row['home_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_own': row['home_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_own': row['home_conceded_official_ewm5'],
            'goals_weighted_official_ewm10_own': row['home_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_own': row['home_conceded_weighted_official_ewm10'],
            
            'goals_official_ewm5_opp': row['away_goals_official_ewm5'],
            'conceded_official_ewm5_opp': row['away_conceded_official_ewm5'],
            'win_rate_official_ewm5_opp': row['away_win_rate_official_ewm5'],
            'goals_official_ewm10_opp': row['away_goals_official_ewm10'],
            'conceded_official_ewm10_opp': row['away_conceded_official_ewm10'],
            'win_rate_official_ewm10_opp': row['away_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_opp': row['away_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_opp': row['away_conceded_weighted_official_ewm5'],
            'goals_weighted_official_ewm10_opp': row['away_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_opp': row['away_conceded_weighted_official_ewm10'],
            
            'was_home': 1 if not row['neutral'] else 0,
            'score': row['home_score'],
            'result_class': home_class
        })
        # アウェイ視点の行
        rows.append({
            'elo_diff': -row['elo_diff'],
            'squad_value_diff': -row['squad_value_diff'],
            'last_wcup_matches_own': row['away_last_wcup_matches'],
            'last_wcup_matches_opp': row['home_last_wcup_matches'],
            
            # 実質ホームアドバンテージ
            'same_conf_own': row['away_same_confederation'],
            'same_conf_opp': row['home_same_confederation'],
            'is_host_own': row['away_is_host'],
            'is_host_opp': row['home_is_host'],
            
            # 基本 rolling
            'goals_roll5_own': row['away_goals_roll5'],
            'conceded_roll5_own': row['away_conceded_roll5'],
            'win_rate_roll5_own': row['away_win_rate_roll5'],
            'goals_roll10_own': row['away_goals_roll10'],
            'conceded_roll10_own': row['away_conceded_roll10'],
            'win_rate_roll10_own': row['away_win_rate_roll10'],
            'goals_weighted_roll5_own': row['away_goals_weighted_roll5'],
            'conceded_weighted_roll5_own': row['away_conceded_weighted_roll5'],
            'goals_weighted_roll10_own': row['away_goals_weighted_roll10'],
            'conceded_weighted_roll10_own': row['away_conceded_weighted_roll10'],
            
            'goals_roll5_opp': row['home_goals_roll5'],
            'conceded_roll5_opp': row['home_conceded_roll5'],
            'win_rate_roll5_opp': row['home_win_rate_roll5'],
            'goals_roll10_opp': row['home_goals_roll10'],
            'conceded_roll10_opp': row['home_conceded_roll10'],
            'win_rate_roll10_opp': row['home_win_rate_roll10'],
            'goals_weighted_roll5_opp': row['home_goals_weighted_roll5'],
            'conceded_weighted_roll5_opp': row['home_conceded_weighted_roll5'],
            'goals_weighted_roll10_opp': row['home_goals_weighted_roll10'],
            'conceded_weighted_roll10_opp': row['home_conceded_weighted_roll10'],

            # 基本 ewm
            'goals_ewm5_own': row['away_goals_ewm5'],
            'conceded_ewm5_own': row['away_conceded_ewm5'],
            'win_rate_ewm5_own': row['away_win_rate_ewm5'],
            'goals_ewm10_own': row['away_goals_ewm10'],
            'conceded_ewm10_own': row['away_conceded_ewm10'],
            'win_rate_ewm10_own': row['away_win_rate_ewm10'],
            'goals_weighted_ewm5_own': row['away_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_own': row['away_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_own': row['away_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_own': row['away_conceded_weighted_ewm10'],
            
            'goals_ewm5_opp': row['home_goals_ewm5'],
            'conceded_ewm5_opp': row['home_conceded_ewm5'],
            'win_rate_ewm5_opp': row['home_win_rate_ewm5'],
            'goals_ewm10_opp': row['home_goals_ewm10'],
            'conceded_ewm10_opp': row['home_conceded_ewm10'],
            'win_rate_ewm10_opp': row['home_win_rate_ewm10'],
            'goals_weighted_ewm5_opp': row['home_goals_weighted_ewm5'],
            'conceded_weighted_ewm5_opp': row['home_conceded_weighted_ewm5'],
            'goals_weighted_ewm10_opp': row['home_goals_weighted_ewm10'],
            'conceded_weighted_ewm10_opp': row['home_conceded_weighted_ewm10'],
            
            # 公式戦優先 rolling
            'goals_official_roll5_own': row['away_goals_official_roll5'],
            'conceded_official_roll5_own': row['away_conceded_official_roll5'],
            'win_rate_official_roll5_own': row['away_win_rate_official_roll5'],
            'goals_official_roll10_own': row['away_goals_official_roll10'],
            'conceded_official_roll10_own': row['away_conceded_official_roll10'],
            'win_rate_official_roll10_own': row['away_win_rate_official_roll10'],
            'goals_weighted_official_roll5_own': row['away_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_own': row['away_conceded_official_roll5'],
            'goals_weighted_official_roll10_own': row['away_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_own': row['away_conceded_official_roll10'],
            
            'goals_official_roll5_opp': row['home_goals_official_roll5'],
            'conceded_official_roll5_opp': row['home_conceded_official_roll5'],
            'win_rate_official_roll5_opp': row['home_win_rate_official_roll5'],
            'goals_official_roll10_opp': row['home_goals_official_roll10'],
            'conceded_official_roll10_opp': row['home_conceded_official_roll10'],
            'win_rate_official_roll10_opp': row['home_win_rate_official_roll10'],
            'goals_weighted_official_roll5_opp': row['home_goals_weighted_official_roll5'],
            'conceded_weighted_official_roll5_opp': row['home_conceded_weighted_official_roll5'],
            'goals_weighted_official_roll10_opp': row['home_goals_weighted_official_roll10'],
            'conceded_weighted_official_roll10_opp': row['home_conceded_weighted_official_roll10'],

            # 公式戦優先 ewm
            'goals_official_ewm5_own': row['away_goals_official_ewm5'],
            'conceded_official_ewm5_own': row['away_conceded_official_ewm5'],
            'win_rate_official_ewm5_own': row['away_win_rate_official_ewm5'],
            'goals_official_ewm10_own': row['away_goals_official_ewm10'],
            'conceded_official_ewm10_own': row['away_conceded_official_ewm10'],
            'win_rate_official_ewm10_own': row['away_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_own': row['away_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_own': row['away_conceded_official_ewm5'],
            'goals_weighted_official_ewm10_own': row['away_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_own': row['away_conceded_weighted_official_ewm10'],
            
            'goals_official_ewm5_opp': row['home_goals_official_ewm5'],
            'conceded_official_ewm5_opp': row['home_conceded_official_ewm5'],
            'win_rate_official_ewm5_opp': row['home_win_rate_official_ewm5'],
            'goals_official_ewm10_opp': row['home_goals_official_ewm10'],
            'conceded_official_ewm10_opp': row['home_conceded_official_ewm10'],
            'win_rate_official_ewm10_opp': row['home_win_rate_official_ewm10'],
            'goals_weighted_official_ewm5_opp': row['home_goals_weighted_official_ewm5'],
            'conceded_weighted_official_ewm5_opp': row['home_conceded_weighted_official_ewm5'],
            'goals_weighted_official_ewm10_opp': row['home_goals_weighted_official_ewm10'],
            'conceded_weighted_official_ewm10_opp': row['home_conceded_weighted_official_ewm10'],
            
            'was_home': 0,
            'score': row['away_score'],
            'result_class': away_class
        })
        
    train_expanded = pd.DataFrame(rows)
    
    # 拡張された特徴量リスト (ewm もすべて追加)
    feature_cols = [
        'elo_diff',
        'squad_value_diff',
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
    
    X = train_expanded[feature_cols]
    y = train_expanded['score']
    y_class = train_expanded['result_class']
    
    # 1. Poisson Regression Model
    print("Training Poisson Regression model with StandardScaler...")
    poisson_pipeline = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=1e-4, max_iter=500)
    )
    poisson_pipeline.fit(X, y)
    
    # 2. LightGBM Poisson Regressor (with Optuna tuning)
    best_reg_params = optimize_lgbm_regressor(train_df, feature_cols, n_trials=30)
    best_reg_params['objective'] = 'poisson'
    best_reg_params['random_state'] = 42
    best_reg_params['verbosity'] = -1
    
    print("Training LightGBM Poisson Regressor with Best Parameters...")
    lgbm_model = LGBMRegressor(**best_reg_params)
    lgbm_model.fit(X, y)
    
    # 3. LightGBM Multiclass Classifier (with Optuna tuning)
    best_cls_params = optimize_lgbm_classifier(train_df, feature_cols, n_trials=30)
    best_cls_params['objective'] = 'multiclass'
    best_cls_params['num_class'] = 3
    best_cls_params['random_state'] = 42
    best_cls_params['verbosity'] = -1
    
    print("Training LightGBM Multiclass Classifier with Best Parameters...")
    lgbm_classifier = LGBMClassifier(**best_cls_params)
    lgbm_classifier.fit(X, y_class)
    
    # 保存
    poisson_path = os.path.join(model_dir, "poisson_model.joblib")
    lgbm_path = os.path.join(model_dir, "lgbm_model.joblib")
    lgbm_classifier_path = os.path.join(model_dir, "lgbm_classifier_model.joblib")
    features_path = os.path.join(model_dir, "feature_cols.joblib")
    
    print(f"\nSaving models to {model_dir}...")
    joblib.dump(poisson_pipeline, poisson_path)
    joblib.dump(lgbm_model, lgbm_path)
    joblib.dump(lgbm_classifier, lgbm_classifier_path)
    joblib.dump(feature_cols, features_path)
    
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()
