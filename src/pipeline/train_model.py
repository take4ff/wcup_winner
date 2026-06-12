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
from features_common import (FEATURE_COLS, expand_df_to_rows,  # noqa: F401
                             CAT_COLS, interleaved_teams, add_team_ids)

# XGBoost分類器の固定パラメータ（walkforwardのensemble検証で採用）
XGB_CLS_PARAMS = {
    'objective': 'multi:softprob', 'num_class': 3, 'n_estimators': 300,
    'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'verbosity': 0,
}

def decay_weights(dates, half_life_years, reference_date=None):
    """試合日からの経過年数に応じた指数減衰重み（試合単位）。half_life<=0 なら等重み"""
    dates = pd.to_datetime(dates)
    if half_life_years is None or half_life_years <= 0:
        return np.ones(len(dates))
    ref = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    age_years = (ref - dates).dt.days / 365.25
    return np.power(0.5, age_years / half_life_years).values


def prepare_cv_folds(train_df, feature_cols, n_splits=5, half_life=0.0):
    """TimeSeriesSplitの各フォールドを展開済みで保持する（トライアル間で共有して再計算を回避）"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(train_df):
        fold_train = train_df.iloc[train_idx]
        expanded_train = expand_df_to_rows(fold_train)
        expanded_val = expand_df_to_rows(train_df.iloc[val_idx])
        # 試合単位の重みを home/away の2行に展開
        w = np.repeat(decay_weights(fold_train['date'], half_life), 2)
        folds.append((
            expanded_train[feature_cols], expanded_train['score'], expanded_train['result_class'],
            expanded_val[feature_cols], expanded_val['score'], expanded_val['result_class'],
            w,
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
        for X_tr, y_tr, _, X_va, y_va, _, w_tr in cv_folds:
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)

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
        for X_tr, _, y_tr_cls, X_va, _, y_va_cls, w_tr in cv_folds:
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr_cls, sample_weight=w_tr)

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
    parser.add_argument('--half_life', type=float, default=0.0,
                        help='時間減衰サンプル重みの半減期(年)。0で無効 (walkforwardで効果検証のうえ設定)')
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

    # 時間減衰サンプル重み（試合単位 → home/away 2行に展開）
    weights = np.repeat(decay_weights(train_df['date'], args.half_life,
                                      reference_date=args.train_end), 2)
    if args.half_life > 0:
        print(f"Applying time-decay sample weights (half_life={args.half_life}y, "
              f"min weight={weights.min():.3f})")

    # 1. Poisson Regression Model
    print("Training Poisson Regression model with StandardScaler...")
    poisson_pipeline = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=1e-4, max_iter=500)
    )
    poisson_pipeline.fit(X, y, poissonregressor__sample_weight=weights)

    # 2. LightGBM Poisson Regressor (with Optuna tuning)
    print("Preparing TimeSeriesSplit CV folds (shared across Optuna trials)...")
    cv_folds = prepare_cv_folds(train_df, feature_cols, n_splits=5, half_life=args.half_life)
    best_reg_params = optimize_lgbm_regressor(cv_folds, n_trials=args.n_trials)
    best_reg_params['objective'] = 'poisson'
    best_reg_params['random_state'] = 42
    best_reg_params['verbosity'] = -1

    print("Training LightGBM Poisson Regressor with Best Parameters...")
    lgbm_model = LGBMRegressor(**best_reg_params)
    lgbm_model.fit(X, y, sample_weight=weights)

    # 3. LightGBM Multiclass Classifier (with Optuna tuning)
    best_cls_params = optimize_lgbm_classifier(cv_folds, n_trials=args.n_trials)
    best_cls_params['objective'] = 'multiclass'
    best_cls_params['num_class'] = 3
    best_cls_params['random_state'] = 42
    best_cls_params['verbosity'] = -1

    print("Training LightGBM Multiclass Classifier with Best Parameters...")
    lgbm_classifier = LGBMClassifier(**best_cls_params)
    lgbm_classifier.fit(X, y_class, sample_weight=weights)

    # 4. チームIDカテゴリカル入り LightGBM Regressor（λアンサンブル第3メンバー）
    #    カテゴリ一覧は全データから固定し、予測時の整合のため保存する
    print("Training team-categorical LightGBM Regressor...")
    team_categories = sorted(set(df['home_team']) | set(df['away_team']))
    own_t, opp_t = interleaved_teams(train_df)
    X_cat = add_team_ids(X, own_t, opp_t, team_categories)
    lgbm_cat = LGBMRegressor(**best_reg_params)
    lgbm_cat.fit(X_cat, y, sample_weight=weights, categorical_feature=CAT_COLS)

    # 5. XGBoost Multiclass Classifier（1X2確率ブレンドの第2分類器）
    print("Training XGBoost Classifier...")
    from xgboost import XGBClassifier
    xgb_classifier = XGBClassifier(**XGB_CLS_PARAMS)
    xgb_classifier.fit(X, y_class, sample_weight=weights)
    
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
    joblib.dump(lgbm_cat, os.path.join(model_dir, "lgbm_cat_model.joblib"))
    joblib.dump(xgb_classifier, os.path.join(model_dir, "xgb_classifier_model.joblib"))
    joblib.dump(team_categories, os.path.join(model_dir, "team_categories.joblib"))
    
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()
