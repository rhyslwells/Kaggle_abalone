# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import scipy
from scipy import stats


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


# Data ===========================================================================================
# ================================================================================================
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
# Summary statistics
print("Summary Statistics:")
print(train.describe())

# Data types and missing values
print("\nData Types:")
print(train.dtypes)
print("\nMissing Values:")
print(train.isnull().sum())

target='Rings'

# EDA visualisation ==============================================================================
# ================================================================================================


# Corrolations ==================================

# Use a clustermap to see the corrolation between the numericals.
numerical_features = train.select_dtypes(include=[np.number])
corr_matrix = numerical_features.corr()
plt.figure(figsize=(12, 10))
sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Clustermap of Correlation between Numerical Features")
plt.show()
#INSIGHTS:
# We see that all features are corrolated roughly similar to the target, and each is correlated with each other close to 1.
# by the denogram we see that Whole_weight_1, Whole_weight_2, Whole_weight, Shell_weight are clustered.
# Similarly are Height, Length and Diameter.

# Distrubution of all numericals =============

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

# Insights:
# 1. Only Height follows a Normal Distribution
# 2. Length & Diameter Features, and Weight Features have similar distribution. 

# Investigate each numerical feature =============

# For each numerical we plot a histogram and a boxplot to see the distrubution of the data, and analyse the spread of the data.

def plot_histogram(data, feature):
    """
    Plot histogram for a numerical feature.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - feature: Name of the numerical feature to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], bins=20, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(data, feature):
    """
    Plot boxplot for a numerical feature.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - feature: Name of the numerical feature to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature], palette='viridis')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.show()

def describe_feature(data, feature):
    """
    Calculate and display summary statistics for a numerical feature.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - feature: Name of the numerical feature to describe.

    Returns:
    - None
    """
    length_description = stats.describe(data[feature])
    length_stats = dict(length_description._asdict())

    # Include in length_stats the data from data[feature].describe()
    length_stats.update(data[feature].describe().to_dict())

    for key, value in length_stats.items():
        print(f"{key}: {value}")


def identify_outliers_by_feature(data, feature):
    """
    Identify outliers for a given feature using the Interquartile Range (IQR) method.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - feature: Name of the numerical feature to identify outliers.

    Returns:
    - outliers: Pandas Series containing the outliers for the specified feature.
    """
    # Calculate the first and third quartiles
    q1 = np.percentile(data[feature], 25)
    q3 = np.percentile(data[feature], 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find outliers
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
    
    return outliers

### Length
feature='Length'

plot_histogram(train, feature)

plot_boxplot(train, feature)

describe_feature(train, feature)

"""
Based on the provided statistical data:
  
- **Variance/Standard Deviation**: The variance is relatively small (0.01398), indicating that the data points are close to the mean on average.
- **Skewness**: The skewness is negative (-0.732), indicating that the distribution is left-skewed. This means that there is a tail towards the lower values, and the bulk of the data is concentrated towards the higher values.
- **Kurtosis**: The kurtosis is positive (0.133), indicating that the distribution is platykurtic. This means that the distribution has thinner tails and a flatter peak compared to a normal distribution.

Overall, the distribution is right-skewed with a wide range of values, a relatively small variance, and a platykurtic shape.
"""

feature_outliers=identify_outliers_by_feature(train, feature)
# remove the outliers from the data and take the boxplot again
train_no_outliers = train[~train[feature].isin(feature_outliers)]
plot_boxplot(train_no_outliers, feature)


### Diameter
### Height
### Whole_weight
### Whole_weight_1
### Whole_weight_2
### Shell_weight



# - mean and median difference implies there are outliers
# - skewness and kurtosis measure the shape of the distribution by looking at the tails of the distribution and the peak of the distribution.



# Handle outliers 

# Log transformation to help the skewness.

# Categoricals and Target analysis ==================================

# Sex is the only categorical feature.

# with a histogram 
vc = train.Rings.value_counts()
plt.figure(figsize=(6, 2))
plt.bar(vc.index, vc)
plt.show()
# Insight: Because all training targets are between 1 and 29, we may clip all predictions to the interval \[1, 29\].

# sns.pairplot

# for each numerical boxplot for sex.

# filter outliers again per sex and numerical?

# Idenifity features which have distinction in heights between boxplots. - apparently none.

# encode to numericals - one hot encoding

# Ask gpt for more things to consider.

# Do statistical tests to see if the difference in means is significant.

# Height as histogram - two overlapping histograms (M & F). 

# Is height a good predictor of sex are these distrubutions statitically different? chi square, kstest, ttest.


## Target Analysis =======================



# Categorical analysis ====================

# Restrict these to the numerical feats we are interested in.
# train.columns
cont_cols = ['Whole_weight','Shell_weight']
sns.pairplot(data=train, vars=cont_cols+[target], hue='Sex', dropna=True)
plt.show()

for col in cont_cols:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='Sex', y=col, data=train)
    plt.title(f'Box plot of {col} by Sex')
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


