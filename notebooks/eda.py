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
train = pd.read_csv(r'..\data\train.csv')
test = pd.read_csv(r'..\data\test.csv')

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
train.shape #(90615, 9)
# Summary statistics
print("Summary Statistics:")
print(train.describe())

# Data types and missing values
print("\nData Types:")
print(train.dtypes)
print("\nMissing Values:")
print(train.isnull().sum())

target='Rings'
# train.columns

# **PROJECT DESCRIPTION**

#Predicting the age of abalone from physical measurements.  The age of abalone is determined counting the number of rings.

# **PHYSICAL ATTRIBUTES**
# 1. **SEX:** <font size="3"> Male/Female/Infant</font>
# 2. **LENGTH:** <font size="3"> Longest shell measurement</font>
# 3. **DIAMETER:** <font size="3"> Diameter of the Abalone</font>
# 4. **HEIGHT:** <font size="3"> Height of the Abalone</font>
# 5. **WHOLE WEIGHT:** <font size="3"> Weight of the whole abalone</font>
# 6. **SHUCKED WEIGHT:** <font size="3"> Weight of the meat</font>
# 7. **VISCERA WEIGHT:** <font size="3"> Gut Weight - Interal Organs</font>
# 8. **SHELL WEIGHT:** <font size="3"> Shell Weight after drying</font>
# 9. **RINGS:** <font size="3"> Number of rings +1.5 gives Age of the Abalone</font>

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
        sns.histplot(train[col], ax=axs[row][col_pos], color='red', kde=True, label='Train', bins=num_bins_train)
        sns.histplot(test[col], ax=axs[row][col_pos], color='blue', kde=True, label='Test', bins=num_bins_test)
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

# For each numerical we plot a histogram and a boxplot to see the distrubution of the data,
# analyse the spread of the data, and identify outliers (Any data point that falls below the lower bound (Q1 - 1.5 * IQR) or above the upper bound (Q3 + 1.5 * IQR) is considered an outlier)

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

    # Find outliers of data as rows
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    
    return outliers

def analyze_numerical_feature(data, feature, plot=True):
    if plot:
        plot_histogram(data, feature)
        print("Boxplot before removing outliers:")
        describe_feature(data, feature)
        plot_boxplot(data, feature)

    feature_outliers = identify_outliers_by_feature(data, feature)

    # remove the outliers from the data and take the boxplot again

    # I want the specific rows of data that are outliers
    data_no_outliers = data[~data[feature].isin(feature_outliers)]
    if plot:
        print("Boxplot after removing outliers:")
        plot_boxplot(data_no_outliers, feature)
    print(f"Percentage which are outliers for {feature}: {len(feature_outliers)/len(data) * 100:.2f}%")

    # train["Height"] = train["Height"].clip(upper=0.5,lower=0.01)
    # test["Height"] = test["Height"].clip(upper=0.5,lower=0.01)  

    return feature_outliers, data_no_outliers

###

### Length

# length_feature_outliers, length_data_no_outliers=analyze_numerical_feature(train, 'Length')
length_feature_outliers, length_data_no_outliers=analyze_numerical_feature(train, 'Length',plot=False)

"""
    #Do analysis using the following template:

    Based on the provided statistical data:
        
    - **Variance/Standard Deviation**: The variance is relatively small (0.01398),
    indicating that the data points are close to the mean on average.
    - **Skewness**: The skewness is negative (-0.732), indicating that the distribution is left-skewed.
        This means that there is a tail towards the lower values, and the bulk of the data is concentrated towards the higher values.
    - **Kurtosis**: The kurtosis is positive (0.133), indicating that the distribution is platykurtic. 
    This means that the distribution has thinner tails and a flatter peak compared to a normal distribution.

    Overall, the distribution is right-skewed with a wide range of values, a relatively small variance, and a platykurtic shape.
"""
### Diameter
# diameter_feature_outliers, diameter_data_no_outliers=analyze_numerical_feature(train, 'Diameter')
diameter_feature_outliers, diameter_data_no_outliers=analyze_numerical_feature(train, 'Diameter',plot=False)

### Height
# height_feature_outliers, height_data_no_outliers=analyze_numerical_feature(train, 'Height')
height_feature_outliers, height_data_no_outliers=analyze_numerical_feature(train, 'Height',plot=False)

### Whole_weight
# whole_weight_feature_outliers, whole_weight_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight')
whole_weight_feature_outliers, whole_weight_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight',plot=False)

### Whole_weight_1
# whole_weight_1_feature_outliers, whole_weight_1_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight_1')
whole_weight_1_feature_outliers, whole_weight_1_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight_1',plot=False)

### Whole_weight_2
# whole_weight_2_feature_outliers, whole_weight_2_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight_2')
whole_weight_2_feature_outliers, whole_weight_2_data_no_outliers=analyze_numerical_feature(train, 'Whole_weight_2',plot=False)

### Shell_weight
# shell_weight_feature_outliers, shell_weight_data_no_outliers=analyze_numerical_feature(train, 'Shell_weight')
shell_weight_feature_outliers, shell_weight_data_no_outliers=analyze_numerical_feature(train, 'Shell_weight',plot=False)

# Handle outliers =============

# Remove all outliers (union)
# Collect all outliers from different features into a set
all_outliers_repeats = concatenated_df = pd.concat([length_feature_outliers,
                             diameter_feature_outliers,
                             height_feature_outliers,
                             whole_weight_feature_outliers,
                             whole_weight_1_feature_outliers,
                             whole_weight_2_feature_outliers,
                             shell_weight_feature_outliers],axis=0)
all_outliers_repeats.shape #(6040,9)

# Remove duplicates from the set of all outliers
all_outliers = all_outliers_repeats.drop_duplicates()
all_outliers.shape #(3349, 9)
# Convert the set of outliers back to a DataFrame
train.shape #(90615, 9)
df_no_outliers = train[~train.isin(all_outliers)].dropna()
df_no_outliers.shape #(87266, 9)

# Remove the outliers from the original data
train=df_no_outliers
train.columns


# Feature Engineering ==================================

train["Volume"] = train["Length"]*train["Diameter"]*train["Height"]
test["Volume"] = test["Length"]*test["Diameter"]*test["Height"] 

train["Density"] = train["Whole_weight"]/train["Volume"]
test["Density"] = test["Whole_weight"]/test["Volume"]

# As we use the RMSLE, all our models will predict the logarithm of the target. 
# In this situation, some models will perform better if we feed them the logarithm of the features. 

# A log transformation is a powerful tool to handle skewed data. It can help to normalize the data
#  and make it more suitable for a machine learning model. This transformation involves taking the logarithm of each data point (y = log(x)).

# num_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in [target]]
# # num_cols

# # log transformation of numericals
# log_features = [f for f in num_cols if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]
# # Log transformation is often applied to right-skewed data to make it more symmetric and closer to a normal distribution.
# log_features #['Whole_weight','Whole_weight_1','Whole_weight_2','Shell_weight','Volume','Density']
# for col in log_features:
#     train[f'log_{col}'] = np.log1p(train[col])
#     test[f'log_{col}'] = np.log1p(test[col])

# # Show histograms of log_Volume and Volume side by side
# plt.figure(figsize=(12, 4))

# # Histogram of log_Volume
# plt.subplot(1, 2, 1)
# sns.histplot(train['log_Volume'], bins=20, kde=True)
# plt.title('Histogram of log_Volume')
# plt.xlabel('log_Volume')

# # Histogram of Volume
# plt.subplot(1, 2, 2)
# sns.histplot(train['Volume'], bins=20, kde=True)
# plt.title('Histogram of Volume')
# plt.xlabel('Volume')

# plt.tight_layout()
# plt.show()

# Do not see much change to normality.


# train.columns

# Categoricals analysis ==================================

# Sex is the only categorical feature.

# Restrict these to the numerical feats we are interested in.
# train.columns # ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']

x_vars = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']
y_vars = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']
sns.pairplot(train, x_vars=x_vars, y_vars=y_vars, hue='Sex', corner=True)

x_vars = ['Whole_weight']
y_vars = ['Length']
sns.pairplot(train, x_vars=x_vars, y_vars=y_vars, hue='Sex', corner=True)

# Insights
# 1. We can see the growth of an Abalone in physical attributes when they mature
# 3. Sex is an important feature especially the immatured category
# 4. Linear correlation across weights is observed and also between length-diameter.

# Aside: Statistical Tests ==================================

# Do Statisical tests to idenifity features which have distinction in heights between boxplots. - apparently none.
# Do statistical tests to see if the difference in means is significant for each feature.
#Compare boxplots first, if noticable difference, then do a statistical test.

# features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']
# for feature in features:
#     # Create a boxplot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x="Sex", y=feature, data=train, order=["I", "M", "F"])
#     plt.title(f"{feature} Distribution by Sex")
#     plt.xlabel("Sex")
#     plt.ylabel(feature)
#     plt.show()

#By inspecting the boxplots for each feature, we can see that Male and Fmale have similar means for each feature.
#However the Infant category has a different means for each feature.

#Let take height (close to normal) as an example and do a statistical test to see if the difference in means is significant, between Infant and Male.
infant_heights = train[train["Sex"] == "I"]["Height"]
male_heights = train[train["Sex"] == "M"]["Height"]

# Perform a t-test for comparing means
t_statistic, p_value = stats.ttest_ind(infant_heights, male_heights)

# Check the significance level
alpha = 0.05
if p_value < alpha:
    print("The difference in means between Infant and Male heights is statistically significant.")
else:
    print("There is no statistically significant difference in means between Infant and Male heights.")


# save train and test
train.to_csv(r'..\data\train_cleaned.csv',index=False)
test.to_csv(r'..\data\test_cleaned.csv',index=False)

