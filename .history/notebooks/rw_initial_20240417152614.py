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

# Q: Why do they have original data?

# Data ==============================

train = pd.read_csv(r'..\train.csv')
test = pd.read_csv(r'..\test.csv')
original = pd.read_csv(r'..\Original.csv')

train_copy=train.copy()
test_copy=test.copy()

# Tag Orignal
original["original"]=1
train["original"]=0
test["original"]=0

train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)
original.drop(columns=["id"],inplace=True)

#possibly combine with train.

train.rename(columns={'Whole weight.1':'Whole_weight_1',
                      'Whole weight.2':'Whole_weight_2',
                      'Whole weight':'Whole_weight',
                      'Shell weight':'Shell_weight'},inplace=True)
test.rename(columns={'Whole weight.1':'Whole_weight_1',
                     'Whole weight.2':'Whole_weight_2',
                     'Whole weight':'Whole_weight',
                     'Shell weight':'Shell_weight'},inplace=True)
original.rename(columns={'Shucked_weight':'Whole_weight_1',
                     'Viscera_weight':'Whole_weight_2'},inplace=True)


train=pd.concat([train,original],axis='rows')

train.head(3)

# Data exploration (before preprocessing) 

# Note:  There is no missing data <- seen

# There is analysis of the target (rings) in notebooks\ps4e4-abalone-age-prediction-regression.ipynb
# with a histogram 

# For some reason, they focus on log transformed histograms.

# Distrubution of numericals

def train_test_distributions():

    cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in [target]]

    # Calculate the number of rows needed for the subplots
    num_rows = (len(cont_cols) + 2) // 3

    # Create subplots for each continuous column
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

    # Loop through each continuous column and plot the histograms
    for i, col in enumerate(cont_cols):
        # Determine the range of values to plot
        max_val = max(train[col].max(), test[col].max(), original[col].max())
        min_val = min(train[col].min(), test[col].min(), original[col].min())
        range_val = max_val - min_val
        
        # Determine the bin size and number of bins
        bin_size = range_val / 20
        num_bins_train = round(range_val / bin_size)
        num_bins_test = round(range_val / bin_size)
        num_bins_original = round(range_val / bin_size)
        
        # Calculate the subplot position
        row = i // 3
        col_pos = i % 3
        
        # Plot the histograms
        sns.histplot(train[col], ax=axs[row][col_pos], color='orange', kde=True, label='Train', bins=num_bins_train)
        sns.histplot(test[col], ax=axs[row][col_pos], color='green', kde=True, label='Test', bins=num_bins_test)
        sns.histplot(original[col], ax=axs[row][col_pos], color='blue', kde=True, label='Original', bins=num_bins_original)
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

# Insights
# 1. Only Height follows a Normal Distribution
# 2. Length Features & Weight Features have similar distribution across them. 
# I suspect strong correlations between these categories. It is natural to have high 
# shell weight & high visceral weight
# ---
# So what- Length and Diameter are close in shape,
#similarly, Whole weight and Shucked weight, viscera weight and shell weight are close in shape.

# Categoricals

# Sex is the only categorical, we plot with numericals
# Restrict these to the numerical feats we are interested in.
sns.pairplot(data=original, vars=cont_cols+[target], hue='Sex')
plt.show()

# Insights

# **INFERENCES**
# 1. We can see the growth of an Abalone in physical attributes when they mature
# 2. There are few datapoints which are outliers because naturally difficult to have low weights, 
# normal length, & 3 times taller than the population(Possibility of experimental errors/noise).
# 3. I think Sex is an important feature especially the immatured category
# 4. Correlation across weights is observed and also between length-diameter.

# trim these

# Preprocessing ==============================

#Handle outliers
# le = LabelEncoder()
# train_data["Sex"] = le.fit_transform(train_data["Sex"])
# test_data["Sex"]  = le.transform(test_data["Sex"])

# train_data["Height"] = train_data["Height"].clip(upper=0.5,lower=0.01)
# test_data["Height"] = test_data["Height"].clip(upper=0.5,lower=0.01)

# log transformation of numericals
# The purpose of this code is to identify features that could 
# potentially benefit from a log transformation.
# A log transformation is a powerful tool to handle skewed data. 
# It can help to normalize the data and make it more suitable for 
# a machine learning model.

log_features = [f for f in num_cols if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]
log_features



#----------

# Feature engineering ==============================

# Model building ==============================

# Model training ==============================
X = train_data.drop(["Rings"],axis=1)
y = train_data["Rings"]

# basic models before hyperparameter tuning

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