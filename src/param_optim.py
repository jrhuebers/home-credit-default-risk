from data_prep import prepare_joined_df
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import time
import gc
from optuna.visualization import plot_optimization_history, plot_param_importances

start = time.time()
train_df = pd.read_csv("data/processed/train.csv")
train_df = train_df.sample(frac=0.1, random_state=42)
gc.collect()
end = time.time()
print(f"time-delta: {end-start}")

def objective(trial):
    start = time.time()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.1),
        "max_depth": trial.suggest_int("max_depth",5,10),
        "reg_alpha": trial.suggest_float("reg_alpha",0,0.3),
        "reg_lambda": trial.suggest_float("reg_lambda",0,1),
        "min_child_weight": trial.suggest_float("min_child_weight",1,50),
        "subsample": trial.suggest_float("subsample",0.5,1),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1)
    }

    model = lgb.LGBMClassifier(
        **params,
        boosting_type="gbdt",
        objective="binary",
        metric="auc",
        silent=-1,
        verbose=-1,
    )

    y = train_df['TARGET']
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X = train_df[feats]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(scores)
    end = time.time()
    print(f"time-delta: {end-start}")

    return scores.mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

print()
print("Best trial:")
print("  Value:", study.best_value)
print("  Params:", study.best_params)

plot_optimization_history(study).show()
plot_param_importances(study).show()

"""
Best trial:
  Value: 0.7764193571743805
  Params: {'n_estimators': 184,
        'learning_rate': 0.049742294234964776,
        'max_depth': 6,
        'reg_alpha': 0.012803654908664254,
        'reg_lambda': 0.9945684681806212,
        'min_child_weight': 43.09702067766131,
        'subsample': 0.6574668923403894,
        'colsample_bytree': 0.5565282153826815}
"""