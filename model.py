import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import xgboost as xgb
import lightgbm as lgb
from data_prep import prepare_application_df, prepare_previous_application_df, prepare_bureau_and_bureau_balance_df, prepare_credit_card_balance_df, prepare_installments_payments_df, prepare_POS_CASH_balance_df


def model_xgboost(train_df, test_df, n_splits):
    print(f"Starting XGBoost. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Cross-validation folds. If n_splits = 1, use the whole dataset
    if n_splits > 1:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=12345)

    # Initialize arrays for storing predictions
    out_of_fold_predictions = np.zeros(train_df.shape[0])
    test_df_predictions = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Iterate over folds
    if n_splits > 1:
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            classifier = xgb.XGBClassifier(
                n_estimators=200,       #set lower than for model_lightgbm
                learning_rate=0.02,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_child_weight=39.3259775,
                subsample=0.8715623,
                colsample_bytree=0.9497036,
                tree_method="hist",  # Use for efficiency
                verbosity=0,
                early_stopping_rounds=40,  # Stops if validation AUC stops improving, set lower than for model_lightgbm
                eval_metric="auc",
            )

            # Define evaluation sets
            evals = [(train_x, train_y), (valid_x, valid_y)]

            # Instead of `callbacks`, use `early_stopping_rounds` and `eval_metric`
            classifier.fit(
                train_x, train_y,
                eval_set=evals,
            )

            # Make predictions
            out_of_fold_predictions[valid_idx] = classifier.predict_proba(valid_x, iteration_range=(0, classifier.best_iteration))[:, 1]
            test_df_predictions += classifier.predict_proba(test_df[feats], iteration_range=(0, classifier.best_iteration))[:, 1] / n_splits

            print(f'Fold {n_fold + 1} AUC: {roc_auc_score(valid_y, out_of_fold_predictions[valid_idx]):.6f}')

            del classifier, train_x, train_y, valid_x, valid_y
            gc.collect()
            
        # Compute AUC
        print(f'AUC score {roc_auc_score(train_df["TARGET"], out_of_fold_predictions):.6f}')

        return(test_df_predictions)
    else:
        train, valid = train_test_split(train_df, test_size=0.2, random_state=12345)
        train_x, train_y = train[feats], train['TARGET']
        valid_x, valid_y = valid[feats], valid['TARGET']


        classifier = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_child_weight=39.3259775,
            subsample=0.8715623,
            colsample_bytree=0.9497036,
            tree_method="hist",  # Use for efficiency
            verbosity=0,
            early_stopping_rounds=200,  # Stops if validation AUC stops improving
            eval_metric="auc",
        )

        # Define evaluation sets
        evals = [(train_x, train_y), (valid_x, valid_y)]

        # Instead of `callbacks`, use `early_stopping_rounds` and `eval_metric`
        classifier.fit(
            train_x, train_y,
            eval_set=evals,
        )

        # Make predictions
        preds = classifier.predict_proba(valid_x, iteration_range=(0, classifier.best_iteration))[:, 1]
        print(f'AUC: {roc_auc_score(valid_y, preds):.6f}')

        test_df_predictions = classifier.predict_proba(test_df[feats])[:, 1]

        del classifier, train_x, train_y, valid_x, valid_y
        gc.collect()

        return test_df_predictions



def model_lightgbm(train_df, test_df, n_splits):
    print(f"Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Cross-validation folds. If n_splits = 1, use the whole dataset
    if n_splits > 1:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=12345)

    # Initialize arrays for storing predictions
    out_of_fold_predictions = np.zeros(train_df.shape[0])
    test_df_predictions = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Iterate over folds
    if n_splits > 1:
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            classifier = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_child_weight=39.3259775,
                subsample=0.8715623,
                colsample_bytree=0.9497036,
                boosting_type="gbdt",
                objective="binary",
                metric="auc",
                silent=-1,
                verbose=-1,
            )

            # Define evaluation sets
            evals = [(train_x, train_y), (valid_x, valid_y)]

            classifier.fit(
                train_x, train_y,
                eval_set=evals,
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
            )

            # Make predictions
            out_of_fold_predictions[valid_idx] = classifier.predict_proba(valid_x)[:, 1]
            test_df_predictions += classifier.predict_proba(test_df[feats])[:, 1] / n_splits

            print(f'Fold {n_fold + 1} AUC: {roc_auc_score(valid_y, out_of_fold_predictions[valid_idx]):.6f}')

            del classifier, train_x, train_y, valid_x, valid_y
            gc.collect()

        # Compute AUC
        print(f'AUC score {roc_auc_score(train_df["TARGET"], out_of_fold_predictions):.6f}')

        return(test_df_predictions)
    else:
        train, valid = train_test_split(train_df, test_size=0.2, random_state=12345)
        train_x, train_y = train[feats], train['TARGET']
        valid_x, valid_y = valid[feats], valid['TARGET']

        classifier = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_child_weight=39.3259775,
            subsample=0.8715623,
            colsample_bytree=0.9497036,
            boosting_type="gbdt",
            objective="binary",
            metric="auc",
            verbose=1,
        )

        # Define evaluation sets
        evals = [(train_x, train_y), (valid_x, valid_y)]

        classifier.fit(
            train_x, train_y,
            eval_set=evals,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True)]
        )

        # Make predictions
        preds = classifier.predict_proba(valid_x)[:, 1]
        print(f'AUC on validation set: {roc_auc_score(valid_y, preds):.6f}')

        test_df_predictions = classifier.predict_proba(test_df[feats])[:, 1]

        del classifier, train_x, train_y, valid_x, valid_y
        gc.collect()

        return(test_df_predictions)




def main():
    train_df = prepare_application_df(train_or_test = "train")
    test_df = prepare_application_df(train_or_test = "test")

    print()
    print("train_df:")
    print(train_df.shape)
    print()
    print("test_df:")
    print(test_df.shape)
    print()

    # merge with bureau_and_bureau_balance_df
    bureau_and_bureau_balance_df = prepare_bureau_and_bureau_balance_df()
    print("bureau_and_bureau_balance_df:")
    print(bureau_and_bureau_balance_df.shape)
    print()
    train_df = train_df.merge(bureau_and_bureau_balance_df, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(bureau_and_bureau_balance_df, on='SK_ID_CURR', how='left')
    del bureau_and_bureau_balance_df
    gc.collect()

    # merge with previous_application_df
    previous_application_df = prepare_previous_application_df()
    print("previous_application_df:")
    print(previous_application_df.shape)
    print()
    train_df = train_df.merge(previous_application_df, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(previous_application_df, on='SK_ID_CURR', how='left')
    del previous_application_df
    gc.collect()

    # merge with installments_payments_df
    installments_payments_df = prepare_installments_payments_df()
    print("installments_payments_df:")
    print(installments_payments_df.shape)
    print()
    train_df = train_df.merge(installments_payments_df, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(installments_payments_df, on='SK_ID_CURR', how='left')
    del installments_payments_df
    gc.collect()

    # merge with POS_CASH_balance_df
    POS_CASH_balance_df = prepare_POS_CASH_balance_df()
    print("POS_CASH_balance_df:")
    print(POS_CASH_balance_df.shape)
    print()
    train_df = train_df.merge(POS_CASH_balance_df, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(POS_CASH_balance_df, on='SK_ID_CURR', how='left')
    del POS_CASH_balance_df
    gc.collect()

    # merge with credit_card_balance_df
    credit_card_balance_df = prepare_credit_card_balance_df()
    print("credit_card_balance_df:")
    print(credit_card_balance_df.shape)
    print()
    train_df = train_df.merge(credit_card_balance_df, on='SK_ID_CURR', how='left')
    test_df = test_df.merge(credit_card_balance_df, on='SK_ID_CURR', how='left')
    del credit_card_balance_df
    gc.collect()

    # Replace special characters (in particular spaces) in column names
    train_df.columns = train_df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    test_df.columns = test_df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    print("train_df:")
    print(train_df.shape)
    print()
    print("test_df:")
    print(test_df.shape)
    print()
    
    train_df = train_df.sample(frac=0.1, random_state=12345)
    gc.collect()

    # k-fold cross-validation. n_splits = 1 for no cross-validation, train-test split 80:20
    n_splits = 1

    # XGBoost (slow):
    #test_df["TARGET"] = model_xgboost(train_df, test_df, n_splits)
    #test_df[["SK_ID_CURR", "TARGET"]].to_csv("submission.csv", index=False)

    # LightGBM (fast):
    test_df["TARGET"] = model_lightgbm(train_df, test_df, n_splits)
    test_df[["SK_ID_CURR", "TARGET"]].to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()