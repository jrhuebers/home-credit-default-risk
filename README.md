# home-credit-default-risk

Gradient boosting learning project. The dataset is from the 2018 Kaggle competition [_Home Credit Default Risk_](https://www.kaggle.com/c/home-credit-default-risk/), predicting the probability of a client defaulting on a loan.

**`data_prep.py`:**
- contains functions for data preprocessing

**`model.py`:**
- function `model_lightgbm`: Implements k-fold cross-validated gradient boosting using LightGBM.
- function `model_xgboost`: Implements k-fold cross-validated gradient boosting using XGBoost.
- function `main`: Calls on the functions from data_prep.py to create the training and testing datasets, then uses either model_lightgbm or model_xgboost to train the model.

The code heavily draws from public Kaggle kernels, especially in terms of feature engineering and choice of hyperparameters. Both `model_lightgbm` and `model_xgboost` can be run without k-fold cross-validation by setting `n_splits = 1`.

For 5-fold cross-validated LightGBM the ROC-AUC score is 0.790376 with the hyperparameters specified in `model.py`.

### E-R diagram of the data
![image](https://github.com/user-attachments/assets/4a078f5f-fd63-4880-8106-c6d5b41dd8b9)
