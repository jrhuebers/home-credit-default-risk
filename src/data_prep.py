
import gc
import numpy as np
import pandas as pd


def prepare_application_df(train_or_test = "train"):
    df = pd.read_csv('data/raw/application_' + train_or_test + '.csv')
    #print('\napplication_' + train_or_test + '_df')
    #print(df.shape)

    # No aggregations because SK_ID_CURR is the primary key already.
    # As we won't aggregate, we also don't need to one-hot encode the categorical columns.
    # Instead, we can just factorize them.
    categorical_columns = [col for col in df.columns if df[col].dtype in ['object','category']]
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])

    # 365.243 should be replaced by NaN
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # add some features:
    # fraction of life employed
    df['DAYS_EMPLOYED_FRAC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # proportion of income to credit
    df['INCOME_CREDIT_FRAC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # income per family member
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # fraction of annual income that is required for loan repayment installments
    df['ANNUITY_INCOME_FRAC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # fraction of credit that is required for loan repayment installments
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df.columns = [''.join(col) for col in df.columns.values]
    return df

def prepare_bureau_and_bureau_balance_df():
    # primary key: SK_BUREAU_ID
    df = pd.read_csv('data/raw/bureau.csv')
    bb_df = pd.read_csv('data/raw/bureau_balance.csv')
    # These two tables are treated together because bureau_balance_df doesn't contain SK_ID_CURR.
    
    #print('\nbureau_and_bureau_balance_df')
    # print(df.shape)
    # print(bb_df.shape)

    # remove outliers?
    
    # one-hot encoding df
    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]
    df[categorical_columns] = df[categorical_columns].fillna('Missing')
    df = pd.get_dummies(df, columns=categorical_columns)
    # print(df.shape)

    # one-hot encode bb_df
    bb_categorical_columns = [col for col in bb_df.columns if bb_df[col].dtype in ['object', 'category']]
    bb_df[bb_categorical_columns] = bb_df[bb_categorical_columns].fillna('Missing')
    bb_df = pd.get_dummies(bb_df, columns=bb_categorical_columns)

    # aggregate by SK_ID_BUREAU, and join to df
    bb_df = bb_df.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['min', 'max', 'size']})
    bb_df.columns = [''.join(col) for col in bb_df.columns.values]
    df = df.join(bb_df, on='SK_ID_BUREAU', how='left')
    df = df.drop(columns=['SK_ID_BUREAU'])

    # Group by SK_ID_CURR, apply aggregate functions.
    # This makes SK_ID_CURR the primary key.
    # Apply mean to the one-hot encoded categorical features:
    categorical_columns = [col for col in df.columns if df[col].dtype == bool]
    categorical_aggregations = {col: 'mean' for col in categorical_columns}

    # Numerical features:
    numerical_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCEmin': ['min'],
        'MONTHS_BALANCEmax': ['max'],
        'MONTHS_BALANCEsize': ['mean', 'sum']
    }
    agg_df = df.groupby('SK_ID_CURR').agg(categorical_aggregations | numerical_aggregations)
    agg_df.columns = [''.join(col) for col in agg_df.columns.values]

    # Now distinguish between active and closed credits!
    # Active credits
    active_df = df[df['CREDIT_ACTIVE_Active'] == 1]
    agg_active_df = active_df.groupby('SK_ID_CURR').agg(numerical_aggregations)
    agg_active_df.columns = pd.Index(['ACTIVE_' + col[0] + "_" + col[1] for col in agg_active_df.columns])
    agg_df = agg_df.join(agg_active_df, how='left', on='SK_ID_CURR')
    del active_df, agg_active_df
    gc.collect()
    # Closed credits
    closed_df = df[df['CREDIT_ACTIVE_Closed'] == 1]
    agg_closed_df = closed_df.groupby('SK_ID_CURR').agg(numerical_aggregations)
    agg_closed_df.columns = pd.Index(['CLOSED_' + col[0] + "_" + col[1] for col in agg_closed_df.columns])
    agg_df = agg_df.join(agg_closed_df, how='left', on='SK_ID_CURR')
    del closed_df, agg_closed_df, df
    gc.collect()

    return agg_df

def prepare_previous_application_df():
    df = pd.read_csv('data/raw/previous_application.csv')
    #print('\nprevious_application_df')
    #print(df.shape)
    # primary key of previous_application_df is SK_ID_PREV rather than SK_ID_CURR

    # what needs to be done:
    # - Days 365.243 replace by NaN
    df['DAYS_FIRST_DRAWING'] = df['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
    df['DAYS_FIRST_DUE'] = df['DAYS_FIRST_DUE'].replace(365243, np.nan)
    df['DAYS_LAST_DUE_1ST_VERSION'] = df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
    df['DAYS_LAST_DUE'] = df['DAYS_LAST_DUE'].replace(365243, np.nan)
    df['DAYS_TERMINATION'] = df['DAYS_TERMINATION'].replace(365243, np.nan)

    # - new feature: proportion value ask to value received

    # - one-hot encoding
    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]
    df[categorical_columns] = df[categorical_columns].fillna('Missing')
    df = pd.get_dummies(df, columns=categorical_columns)
    #print(df.shape)

    # - group by SK_ID_CURR, apply aggregate functions
    categorical_columns = [col for col in df.columns if df[col].dtype == bool]
    categorical_aggregations = {col: 'mean' for col in categorical_columns}
    # - - Numerical features
    numerical_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    agg_df = df.groupby('SK_ID_CURR').agg(categorical_aggregations)

    # - - Distinguish between refused and approved applications:
    # - - - Approved applications
    approved_df = df[df['NAME_CONTRACT_STATUS_Approved'] == 1]
    agg_approved_df = approved_df.groupby('SK_ID_CURR').agg(numerical_aggregations)
    agg_approved_df.columns = pd.Index(['APPROVED_' + col[0] + "_" + col[1] for col in agg_approved_df.columns])
    agg_df = agg_df.join(agg_approved_df, how='left', on='SK_ID_CURR')
    del approved_df, agg_approved_df
    gc.collect()
    # - - - Refused applications
    refused_df = df[df['NAME_CONTRACT_STATUS_Refused'] == 1]
    agg_refused_df = refused_df.groupby('SK_ID_CURR').agg(numerical_aggregations)
    agg_refused_df.columns = pd.Index(['REFUSED_' + col[0] + "_" + col[1] for col in agg_refused_df.columns])
    agg_df = agg_df.join(agg_refused_df, how='left', on='SK_ID_CURR')
    del refused_df, agg_refused_df, df
    gc.collect()

    return agg_df

def prepare_POS_CASH_balance_df():
    # categorical columns: ['NAME_CONTRACT_STATUS']
    # numerical columns: ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
    #                           'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF']
    # no primary key. df will be joined to application_train_test_df on SK_ID_CURR.

    df = pd.read_csv('data/raw/POS_CASH_balance.csv')
    # print("\nPOS_CASH_balance_df")
    # print(df.shape)

    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]

    df[categorical_columns] = df[categorical_columns].fillna('Missing')
    df = pd.get_dummies(df, columns=categorical_columns)

    # aggregate by SK_ID_CURR
    categorical_columns = [col for col in df.columns if df[col].dtype == bool]
    cat_agg_dict = {col: 'mean' for col in categorical_columns}
    agg_dict = {
        'MONTHS_BALANCE': ['size', 'mean', 'max'],
        'CNT_INSTALMENT': ['mean', 'max', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'sum'],
        'SK_DPD': ['mean', 'max'],
        'SK_DPD_DEF': ['mean', 'max']
    }
    agg_dict |= cat_agg_dict
    df = df.groupby('SK_ID_CURR').agg(agg_dict)

    gc.collect()

    df.columns = [''.join(col) for col in df.columns.values]
    return df


def prepare_installments_payments_df():
    # categorical columns: []
    # numerical columns: ['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER',
    #                       'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT',
    #                       'PAYMENT_FRAC', 'PAYMENT_DIFF', 'DPD', 'DBD']
    # no primary key. df will be joined to application_train_test_df on SK_ID_CURR.

    df = pd.read_csv('data/raw/installments_payments.csv')
    # print("\ninstallments_payments_df")
    # print(df.shape)

    # Proportion and difference paid in each installment (amount paid and installment value)
    df['PAYMENT_FRAC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
    df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)

    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]

    df[categorical_columns] = df[categorical_columns].fillna('Missing')
    df = pd.get_dummies(df, columns=categorical_columns)

    # aggregate by SK_ID_CURR
    categorical_columns = [col for col in df.columns if df[col].dtype == bool]
    cat_agg_dict = {col: 'mean' for col in categorical_columns}
    agg_dict = {col: ['min', 'mean', 'max', 'sum', 'var'] for col in df.columns if col not in categorical_columns + ['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']}
    agg_dict |= cat_agg_dict | {'NUM_INSTALMENT_VERSION': ['nunique']}
    df = df.groupby('SK_ID_CURR').agg(agg_dict)
    df.columns = ['_'.join(col) for col in df.columns.values]
    
    gc.collect()

    df.columns = [''.join(col) for col in df.columns.values]
    return df

def prepare_credit_card_balance_df():
    # categorical columns: ['NAME_CONTRACT_STATUS']
    # numerical columns: ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
    #       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
    #       'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
    #       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
    #       'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT',
    #       'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
    #       'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF']
    # no primary key. df will be joined to application_train_test_df on SK_ID_CURR.

    df = pd.read_csv('data/raw/credit_card_balance.csv')
    #print("\ncredit_card_balance_df")
    #print(df.shape)

    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]

    df[categorical_columns] = df[categorical_columns].fillna('Missing')
    df = pd.get_dummies(df, columns=categorical_columns)

    # aggregate by SK_ID_CURR
    categorical_columns = [col for col in df.columns if df[col].dtype == bool]
    cat_agg_dict = {col: 'mean' for col in categorical_columns}
    agg_dict = {col: ['min', 'mean', 'max', 'sum', 'var'] for col in df.columns if col not in categorical_columns + ['SK_ID_PREV', 'SK_ID_CURR']}
    agg_dict |= cat_agg_dict
    df = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    gc.collect()

    df.columns = [''.join(col) for col in df.columns.values]
    return df



def prepare_joined_df():
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

    return train_df, test_df



def main():
    """
    application_train_df = prepare_application_df(train_or_test = "train")
    print("application_train_df")
    # print(application_train_df.head())
    # application_train_df.info()

    application_test_df = prepare_application_df(train_or_test = "test")
    print("application_test_df")
    # print(application_test_df.head())
    # application_test_df.info()

    bureau_and_bureau_balance_df = prepare_bureau_and_bureau_balance_df()
    print("bureau_and_bureau_balance_df")
    # print(bureau_and_bureau_balance_df.head())
    # bureau_and_bureau_balance_df.info()

    previous_application_df = prepare_previous_application_df()
    print("previous_application_df")
    # print(previous_application_df.head())
    # previous_application_df.info()

    POS_CASH_balance_df = prepare_POS_CASH_balance_df()
    print("POS_CASH_balance_df")
    # print(POS_CASH_balance_df.head())
    # POS_CASH_balance_df.info()

    installments_payments_df = prepare_installments_payments_df()
    print("installments_payments_df")
    # print(installments_payments_df.head())
    # installments_payments_df.info()

    credit_card_balance_df = prepare_credit_card_balance_df()
    print("credit_card_balance_df")
    # print(credit_card_balance_df.head())
    # credit_card_balance_df.info()
    """

    train_df, test_df = prepare_joined_df()
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == '__main__':
    main()