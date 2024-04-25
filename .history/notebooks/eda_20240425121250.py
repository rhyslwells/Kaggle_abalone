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

# **PROJECT DESCRIPTION**

# <font size="3">Predicting the age of abalone from physical measurements.  The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope</font>

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

# 3. **Density**:<font size="3">Measure of body density</font>



train["Volume"] = train["Length"]*train["Diameter"]*train["Height"]
test["Volume"] = test["Length"]*test["Diameter"]*test["Height"] 

train["Density"] = train["Whole_weight"]/train["Volume"]
test["Density"] = test["Whole_weight"]/test["Volume"]

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

# <font size="3">We're going to see what transformation works better for each feature and select them, the idea is to compress the data. There could be situations where you will have to stretch the data. These are the methods applied:</font>

# 1. **Log Transformation**: <font size="3">This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = log(x)
# why log?
# train["Volume_log"] = np.log(train["Volume"])
# test["Volume_log"] = np.log(test["Volume"])



train.columns

# Categoricals analysis ==================================

# Sex is the only categorical feature.

# Restrict these to the numerical feats we are interested in.
# train.columns
# ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']

x_vars = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']
y_vars = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Whole_weight_1','Whole_weight_2', 'Shell_weight']
sns.pairplot(train, x_vars=x_vars, y_vars=y_vars, hue='Sex', corner=True)

x_vars = ['Whole_weight']
y_vars = ['Length']
sns.pairplot(train, x_vars=x_vars, y_vars=y_vars, hue='Sex', corner=True)

# Insights
# 1. We can see the growth of an Abalone in physical attributes when they mature
# 3. I think Sex is an important feature especially the immatured category
# 4. Linear correlation across weights is observed and also between length-diameter.

# for each numerical get a boxplot for sex and possibly filter outliers again.

# Do Statisical tests to idenifity features which have distinction in heights between boxplots. - apparently none.

# Do statistical tests to see if the difference in means is significant.
# Height as histogram - two overlapping histograms (M & F). 
# Is height a good predictor of sex are these distrubutions statitically different? chi square, kstest, ttest.



# Feature Elimination ==================================

# **Steps to Eliminate Correlated  Features**:
# 1. <font size="3">Group features based on their parent feature. For example, all features derived from weight come under one set</font>
# 2. <font size="3">Apply PCA on the set, Cluster-Target Encoding on the set</font>
# 3. <font size="3">See the performance of each feature on a cross-validated single feature-target model</font>
# 4. <font size="3">Select the feature with highest CV-MAE</font>


# first_drop=[ f for f in unimportant_features if f in train.columns]
# train=train.drop(columns=first_drop)
# test=test.drop(columns=first_drop)

# final_drop_list=[]

# table = PrettyTable()
# table.field_names = ['Original', 'Final Transformation', "RMSLE(CV)- Regression"]
# dt_params={'criterion': 'absolute_error'}
# threshold=0.85
# # It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
# best_cols=[]

# for col in cont_cols:
#     sub_set=[f for f in train.columns if col in f and train[f].nunique()>100]
#     print(sub_set)
#     if len(sub_set)>2:
#         correlated_features = []

#         for i, feature in enumerate(sub_set):
#             # Check correlation with all remaining features
#             for j in range(i+1, len(sub_set)):
#                 correlation = np.abs(train[feature].corr(train[sub_set[j]]))
#                 # If correlation is greater than threshold, add to list of highly correlated features
#                 if correlation > threshold:
#                     correlated_features.append(sub_set[j])

#         # Remove duplicate features from the list
#         correlated_features = list(set(correlated_features))
#         print(correlated_features)
#         if len(correlated_features)>=2:

#             temp_train=train[correlated_features]
#             temp_test=test[correlated_features]
#             #Scale before applying PCA
#             sc=StandardScaler()
#             temp_train=sc.fit_transform(temp_train)
#             temp_test=sc.transform(temp_test)

#             # Initiate PCA
#             pca=TruncatedSVD(n_components=1)
#             x_pca_train=pca.fit_transform(temp_train)
#             x_pca_test=pca.transform(temp_test)
#             x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
#             x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
#             train=pd.concat([train,x_pca_train],axis='columns')
#             test=pd.concat([test,x_pca_test],axis='columns')

#             # Clustering
#             model = KMeans()
#             kmeans = KMeans(n_clusters=28)
#             kmeans.fit(np.array(temp_train))
#             labels_train = kmeans.labels_

#             train[col+'_final_cluster'] = labels_train
#             test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

#             cat_labels=cat_labels=train.groupby([col+"_final_cluster"])[target].mean()
#             cat_labels2=cat_labels.to_dict()
#             train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
#             test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

#             correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

#             # See which transformation along with the original is giving you the best univariate fit with target
#             kf=KFold(n_splits=5, shuffle=True, random_state=42)

#             rmse_scores = []

#             for f in temp_cols:
#                 X = train_copy[[f]].values
#                 y = train_copy[target].astype(int).values

#                 rmses = []
#                 for train_idx, val_idx in kf.split(X, y):
#                     X_train, y_train = X[train_idx], y[train_idx]
#                     x_val, y_val = X[val_idx], y[val_idx]
#                     model=LinearRegression()
#                     model.fit(X_train,np.log1p(y_train))
#                     y_pred=nearest(np.expm1(model.predict(x_val)))
#                     rmses.append(rmse(np.log1p(y_val),np.log1p(y_pred)))
                    
#                 if f not in best_cols:
#                     rmse_scores.append((f,np.mean(rmses)))
#             best_col, best_rmse=sorted(rmse_scores, key=lambda x:x[1], reverse=False)[0]
#             best_cols.append(best_col)

#             cols_to_drop = [f for f in correlated_features if  f not in best_cols]
#             if cols_to_drop:
#                 final_drop_list=final_drop_list+cols_to_drop
#             table.add_row([col,best_col ,best_acc])

# print(table)      

#features selection ==================================



# final_features=[f for f in train.columns if f not in [target]]
# final_features=[*set(final_features)]

# sc=StandardScaler()

# train_scaled=train.copy()
# test_scaled=test.copy()
# train_scaled[final_features]=sc.fit_transform(train[final_features])
# test_scaled[final_features]=sc.transform(test[final_features])
# len(final_features)

# def post_processor(train, test):
#     cols=train.drop(columns=[target]).columns
#     train_cop=train.copy()
#     test_cop=test.copy()
#     drop_cols=[]
#     for i, feature in enumerate(cols):
#         for j in range(i+1, len(cols)):
#             if sum(abs(train_cop[feature]-train_cop[cols[j]]))==0:
#                 if cols[j] not in drop_cols:
#                     drop_cols.append(cols[j])
#     print(drop_cols)
#     train_cop.drop(columns=drop_cols,inplace=True)
#     test_cop.drop(columns=drop_cols,inplace=True)
    
#     return train_cop, test_cop
                    
# train, test=   post_processor(train_scaled, test_scaled)