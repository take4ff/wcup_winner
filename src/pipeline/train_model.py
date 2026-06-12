import os
import argparse
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

# 学習・バックテスト・2026年予測で共通の特徴量リスト
# 特徴量定義と own/opp 展開は features_common.py に一元化されている
from features_common import FEATURE_COLS, expand_df_to_rows  # noqa: F401

def prepare_cv_folds(train_df, feature_cols, n_splits=5):
    """TimeSeriesSplitの各フォールドを展開済みで保持する（トライアル間で共有して再計算を回避）"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(train_df):
        expanded_train = expand_df_to_rows(train_df.iloc[train_idx])
        expanded_val = expand_df_to_rows(train_df.iloc[val_idx])
        folds.append((
            expanded_train[feature_cols], expanded_train['score'], expanded_train['result_class'],
            expanded_val[feature_cols], expanded_val['score'], expanded_val['result_class'],
        ))
    return folds

def optimize_lgbm_regressor(cv_folds, n_trials=30):
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
        losses = []
        for X_tr, y_tr, _, X_va, y_va, _ in cv_folds:
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_va)
            preds = np.clip(preds, 1e-10, None)
            loss = np.mean(preds - y_va * np.log(preds))
            losses.append(loss)

        return np.mean(losses)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LGBMRegressor parameters: {study.best_params}")
    return study.best_params

def optimize_lgbm_classifier(cv_folds, n_trials=30):
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
        losses = []
        for X_tr, _, y_tr_cls, X_va, _, y_va_cls in cv_folds:
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr_cls)

            proba = model.predict_proba(X_va)
            epsilon = 1e-15
            proba = np.clip(proba, epsilon, 1 - epsilon)

            one_hot = np.eye(3)[y_va_cls]
            loss = -np.mean(np.sum(one_hot * np.log(proba), axis=1))
            losses.append(loss)

        return np.mean(losses)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LGBMClassifier parameters: {study.best_params}")
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description='Poisson + LightGBM アンサンブルモデル学習')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='学習データの開始日 (default: 2015-01-01)')
    parser.add_argument('--train_end', type=str, default=None,
                        help='学習データの終了日(その日を含まない)。バックテスト用に 2022-11-20 等を指定。省略時は全データで学習')
    parser.add_argument('--n_trials', type=int, default=30, help='Optunaのトライアル数 (default: 30)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='モデル保存先 (base_dirからの相対パス)。バックテスト用は models/backtest_2022 等を指定して本番モデルと分離する (default: models)')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_path = os.path.join(base_dir, "data/processed/features.csv")
    model_dir = os.path.join(base_dir, args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model output dir: {model_dir}")

    print(f"Reading features from {input_path}...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    # 学習データの期間設定
    train_df = df[df['date'] >= args.train_start].copy()
    if args.train_end:
        train_df = train_df[train_df['date'] < args.train_end]
    print(f"Training period: {args.train_start} .. {args.train_end or 'latest'}")
    print(f"Number of training matches: {len(train_df)}")
    
    # データをチーム視点（対称）に展開して学習データを構築
    train_expanded = expand_df_to_rows(train_df)
    feature_cols = FEATURE_COLS
    
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
    print("Preparing TimeSeriesSplit CV folds (shared across Optuna trials)...")
    cv_folds = prepare_cv_folds(train_df, feature_cols, n_splits=5)
    best_reg_params = optimize_lgbm_regressor(cv_folds, n_trials=args.n_trials)
    best_reg_params['objective'] = 'poisson'
    best_reg_params['random_state'] = 42
    best_reg_params['verbosity'] = -1
    
    print("Training LightGBM Poisson Regressor with Best Parameters...")
    lgbm_model = LGBMRegressor(**best_reg_params)
    lgbm_model.fit(X, y)
    
    # 3. LightGBM Multiclass Classifier (with Optuna tuning)
    best_cls_params = optimize_lgbm_classifier(cv_folds, n_trials=args.n_trials)
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
