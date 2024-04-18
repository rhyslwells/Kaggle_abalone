# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import scipy

from sklearn.compose import TransformedTargetRegressor
from sklearn import set_config
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder, OneHotEncoder, MEstimateEncoder, OrdinalEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer

from catboost import CatBoostRegressor

sns.set_theme(style = 'white', palette = 'colorblind')
pal = sns.color_palette('colorblind')

pd.set_option('display.max_rows', 100)
set_config(transform_output = 'pandas')
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


# Data ==============================

train = pd.read_csv(r'..\train.csv')
test = pd.read_csv(r'..\test.csv')

train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)

train.rename(columns={'Whole weight.1':'Whole_weight_1',
                      'Whole weight.2':'Whole_weight_2',
                      'Whole weight':'Whole_weight',
                      'Shell weight':'Shell_weight'},inplace=True)
test.rename(columns={'Whole weight.1':'Whole_weight_1',
                     'Whole weight.2':'Whole_weight_2',
                     'Whole weight':'Whole_weight',
                     'Shell weight':'Shell_weight'},inplace=True)

train.head(3)
# Check for missing data
train.isnull().sum()
# Note:  There is no missing data <- seen
target='Rings'

# EDA ============================================================================================
# ================================================================================================



## Target Analysis =======================

# with a histogram 
vc = train.Rings.value_counts()
plt.figure(figsize=(6, 2))
plt.bar(vc.index, vc)
plt.show()
# Insight: Because all training targets are between 1 and 29, we may clip all predictions to the interval \[1, 29\].


# Distrubution of numericals =============

def train_test_distributions(target='Rings'):
    """
    Plot the histograms of continuous columns in the train and test datasets.

    This function calculates the number of rows needed for the subplots based on the number of continuous columns.
    It then creates subplots for each continuous column and plots the histograms of the train and test datasets.

    Returns:
        None
    """
    cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in [target]]

    # Calculate the number of rows needed for the subplots
    num_rows = (len(cont_cols) + 2) // 3

    # Create subplots for each continuous column
    _, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

    # Loop through each continuous column and plot the histograms
    for i, col in enumerate(cont_cols):
        # Determine the range of values to plot
        max_val = max(train[col].max(), test[col].max())
        min_val = min(train[col].min(), test[col].min())
        range_val = max_val - min_val
        
        # Determine the bin size and number of bins
        bin_size = range_val / 20
        num_bins_train = round(range_val / bin_size)
        num_bins_test = round(range_val / bin_size)
        
        # Calculate the subplot position
        row = i // 3
        col_pos = i % 3
        
        # Plot the histograms
        sns.histplot(train[col], ax=axs[row][col_pos], color='orange', kde=True, label='Train', bins=num_bins_train)
        sns.histplot(test[col], ax=axs[row][col_pos], color='green', kde=True, label='Test', bins=num_bins_test)
        axs[row][col_pos].set_title(col)
        axs[row][col_pos].set_xlabel('Value')
        axs[row][col_pos].set_ylabel('Frequency')
        axs[row][col_pos].legend()

    # Remove any empty subplots
    if len(cont_cols) % 3 != 0:
        for col_pos in range(len(cont_cols) % 3, 3):
            axs[-1][col_pos].remove()

    plt.tight_layout()
    plt.show()
train_test_distributions()

print("Insights:\n"
    "1. Only Height follows a Normal Distribution\n"
    "2. Length Features & Weight Features have similar distribution across them. \n"
    "   I suspect strong correlations between these categories. It is natural to have high\n"
    "   shell weight & high visceral weight\n"
    "---\n"
    "So what- Length and Diameter are close in shape,\n"
    "   similarly, Whole weight and Shucked weight, viscera weight and shell weight are close in shape.")

# Categorical analysis ====================

# Sex is the only categorical, we plot with numericals

# Restrict these to the numerical feats we are interested in.
# train.columns
cont_cols = ['Whole_weight','Shell_weight']
sns.pairplot(data=train, vars=cont_cols+[target], hue='Sex', dropna=True)
plt.show()


# Insights
# 1. We can see the growth of an Abalone in physical attributes when they mature
# 2. There are few datapoints which are outliers because naturally difficult to have low weights, 
# normal length, & 3 times taller than the population(Possibility of experimental errors/noise).
# 3. I think Sex is an important feature especially the immatured category
# 4. Correlation across weights is observed and also between length-diameter.

# corrolation plot <- what does it say, clustermap?
cc = np.corrcoef(train[numeric_vars], rowvar=False)
sns.heatmap(cc, center=0, cmap='coolwarm', annot=True,
            xticklabels=numeric_vars, yticklabels=numeric_vars)
plt.show()
# Tells us all features are improtant. No need to drop.

# Preprocessing ==============================

#Handle outliers
# le = LabelEncoder()
# train_data["Sex"] = le.fit_transform(train_data["Sex"])
# test_data["Sex"]  = le.transform(test_data["Sex"])

# train_data["Height"] = train_data["Height"].clip(upper=0.5,lower=0.01)
# test_data["Height"] = test_data["Height"].clip(upper=0.5,lower=0.01)

# log transformation of numericals
log_features = [f for f in num_cols if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]
log_features
log_features = []
for col in numeric_features:
    train[f'log_{col}'] = np.log1p(train[col])
    test[f'log_{col}'] = np.log1p(test[col])
    log_features.append(f'log_{col}')
# The purpose of this code is to identify features that could 
# potentially benefit from a log transformation.
# A log transformation is a powerful tool to handle skewed data. 
# It can help to normalize the data and make it more suitable for 
# a machine learning model.


# possibley after feature eengineering




#----------

# Feature engineering ==============================

# Basic calculations see eda notebooks

# Model building ==============================
from sklearn.metrics import mean_squared_log_error, make_scorer

# Define a function for RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Create a scorer for RMSLE
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lgbm', LGBMRegressor(objective='regression'))
])

# Parameters grid to search
param_grid = {
    'lgbm__num_leaves': [31, 50, 100],
    'lgbm__learning_rate': [0.05, 0.1],
    'lgbm__n_estimators': [1500]
}

# Setup the grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=rmsle_scorer, verbose=1)

# Run grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model using RMSLE
rmsle_value = rmsle(y_test, y_pred)
print(f'Root Mean Squared Logarithmic Error of the best model: {rmsle_value}')

# - `Sex` is a categorical feature. For some models, we'll need to one-hot encode it, for other models it suffices to mark it as categorical.



# Model training ==============================
X = train_data.drop(["Rings"],axis=1)
y = train_data["Rings"]



# basic models before hyperparameter tuning

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def cross_validate(model, label, features=test.columns, n_repeats=1):
    """
    
    
    # Cross-validation

    To ensure that our cross-validation results are consistent, we'll use the same function for cross-validating all models.

    Notice that in cross-validation, we first split the dataset and then add the original data only 
    to the training dataset. The validation dataset consists purely of competition data. 
    This setup lets us correctly assess whether the original data are useful or harmful.
    

    Compute out-of-fold and test predictions for a given model.
    
    Out-of-fold and test predictions are stored in the global variables
    oof and test_pred, respectively.
    
    If n_repeats > 1, the model is trained several times with different seeds.
    
    All predictions are clipped to the interval [1, 29].
    """
    scores = []
    oof_preds = np.full_like(train.Rings, np.nan, dtype=float)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.Rings)):
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train.iloc[idx_tr].Rings
        y_va = train.iloc[idx_va].Rings
        
        if USE_ORIGINAL_DATA:
            X_tr = pd.concat([X_tr, original_dataset[features]], axis=0)
            y_tr = pd.concat([y_tr, original_dataset.Rings], axis=0)
            
        y_pred = np.zeros_like(y_va, dtype=float)
        for i in range(n_repeats):
            m = clone(model)
            if n_repeats > 1:
                mm = m
                if isinstance(mm, Pipeline):
                    mm = mm[-1]
                if isinstance(mm, TransformedTargetRegressor):
                    mm = mm.regressor
                mm.set_params(random_state=i)
            m.fit(X_tr, y_tr)
            y_pred += m.predict(X_va)
        y_pred /= n_repeats
        y_pred = y_pred.clip(1, 29)
        
#         residuals = np.log1p(y_va) - np.log1p(y_pred)
#         plt.figure(figsize=(6, 2))
#         plt.scatter(y_pred, residuals, s=1)
#         plt.axhline(0, color='k')
#         plt.show()
        
        score = mean_squared_log_error(y_va, y_pred, squared=False)
        print(f"# Fold {fold}: RMSLE={score:.5f}")
        scores.append(score)
        oof_preds[idx_va] = y_pred
    print(f"{Fore.GREEN}# Overall: {np.array(scores).mean():.5f} {label}{Style.RESET_ALL}")
    oof[label] = oof_preds
    
    if COMPUTE_TEST_PRED:
        # Retrain n_repeats times with the whole dataset and average
        y_pred = np.zeros(len(test), dtype=float)
        X_tr = train[features]
        y_tr = train.Rings
        if USE_ORIGINAL_DATA:
            X_tr = pd.concat([X_tr, original_dataset[features]], axis=0)
            y_tr = pd.concat([y_tr, original_dataset.Rings], axis=0)
        for i in range(n_repeats):
            m = clone(model)
            if n_repeats > 1:
                mm = m
                if isinstance(mm, Pipeline):
                    mm = mm[-1]
                if isinstance(mm, TransformedTargetRegressor):
                    mm = mm.regressor
                mm.set_params(random_state=i)
            m.fit(X_tr, y_tr)
            y_pred += m.predict(test[features])
        y_pred /= n_repeats
        y_pred = y_pred.clip(1, 29)
        test_pred[label] = y_pred


# PolynomialFeatures + Ridge (linear model) one-hot-encode categroical
model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
                                        remainder='passthrough'),
                      StandardScaler(),
                      PolynomialFeatures(degree=3),
                      TransformedTargetRegressor(Ridge(100),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1))
cross_validate(model, 'Poly-Ridge', numeric_features + log_features + ['Sex'])
# Overall: 0.15293 Poly-Ridge

# Random forest
model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
                                        remainder='passthrough'),
                      TransformedTargetRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=5),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1))
cross_validate(model, 'Random forest', log_features + ['Sex'])
# Overall: 0.14962 Random forest

lgbmmodel = LGBMRegressor(random_state=seed, verbose=-1)

# xgbmodel = XGBRegressor(random_state=seed)
# catmodel = CatBoostRegressor(random_state=seed, verbose=0)

print("CV RMSLE score of LGBM is ",np.sqrt(-cross_val_score(lgbmmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))
# print("CV RMSLE score of XGB is ",np.sqrt(-cross_val_score(xgbmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))
# print("CV RMSLE score of CAT is ",np.sqrt(-cross_val_score(catmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))

# Refining hyperparameters using optuna

lgbm_params = {
    'n_estimators' : 15000,  
    "random_state": seed,
    "boosting_type": "gbdt",    
    "objective":'regression',
    "device": "gpu",
    "verbose": -1,
    "early_stopping_rounds" : 4000,
    'max_depth': 9,
    'learning_rate': 0.0754689136929529,
    'min_child_weight': 2.9774820924588674,
    'min_child_samples': 172,
    'subsample': 0.749283862376052,
    'subsample_freq': 0,
    'colsample_bytree': 0.5668465666039963,
    'num_leaves': 18,
    'lambda_l1': 4.011146777594568e-05,
    'lambda_l2': 0.18342984449081373,
    'metric': 'huber'
}
SPLITS = 5
REPEATS = 1
lgbm_score = []

#use RepeatedStratifiedKFold

for i,(tr,val) in enumerate(RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS,random_state=seed).split(X,y)):
    
    print("-"*30,f"FOLD {i+1}/{SPLITS*REPEATS}","-"*30)
    X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]
    
    print("\n->","LGBM:")
    lgbmmodel = LGBMRegressor(**lgbm_params)
    lgbmmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)], eval_names=["valid"],eval_metric=['MSLE'])
    msle = mean_squared_log_error(y_test, lgbmmodel.predict(X_test))
    rmsle = np.sqrt(msle)
    lgbm_score.append(rmsle)
    print(f"Fold {i+1} RMSLE of LGBM =", rmsle,"\n")
    submission["Rings"] += lgbm_wt*lgbmmodel.predict(test_data)
    

# With the same data sets compare models trained againest each other.

# Submission