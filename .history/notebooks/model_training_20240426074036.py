import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

# TODO: test models for different hyperparameters and plot the results for each model.


# Load the data ==============================
train = pd.read_csv(r'..\data\train_cleaned.csv')
test = pd.read_csv(r'..\data\test_cleaned.csv')

# Encode categorical variables to numericals
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

# Split features and target variable
X = train.drop(columns=['Rings'])
y = train['Rings']

# Baseline Models ==============================
seed = 123

lgbmmodel = LGBMRegressor(random_state=seed, verbose=-1)
lgbm_rmsle = np.sqrt(-cross_val_score(lgbmmodel, X, y, cv=4, scoring='neg_mean_squared_log_error').mean())
print("CV RMSLE score of LGBM is ", lgbm_rmsle)
# CV RMSLE score of LGBM is  0.15021509945839864

# XGBRegressor
xgbmodel = XGBRegressor(random_state=seed)
xgb_rmsle = np.sqrt(-cross_val_score(xgbmodel, X, y, cv=4, scoring='neg_mean_squared_log_error').mean())
print("CV RMSLE score of XGB is ", xgb_rmsle)
# CV RMSLE score of XGB is  0.15131278618955712

# CatBoostRegressor
catmodel = CatBoostRegressor(random_state=seed, verbose=0)
cat_rmsle = np.sqrt(-cross_val_score(catmodel, X, y, cv=4, scoring='neg_mean_squared_log_error').mean())
print("CV RMSLE score of CAT is ", cat_rmsle)
# CV RMSLE score of CAT is  0.14926420788403344


# Cross-validation ==============================

# To ensure that our cross-validation results are consistent, we'll use the same function for cross-validating all models.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def cross_validate(model, label, features=test.columns, n_repeats=1):
    """Compute out-of-fold and test predictions for a given model.
    
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

# Baseline Models 2 ==============================

# log_features
# numeric_features

# Random forest
model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
                                        remainder='passthrough'),
                      TransformedTargetRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=5),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1))
cross_validate(model, 'Random forest', log_features + ['Sex'])
# Overall: 0.14962 Random forest


# LightGBM
# Hyperparameters were tuned with Optuna
lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.038622511348472645, 'colsample_bytree': 0.5757189042456357, 'reg_lambda': 0.09664116733307193, 'min_child_samples': 87, 'num_leaves': 43, 'verbose': -1} # 0.14804
model = TransformedTargetRegressor(lightgbm.LGBMRegressor(**lgbm_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'LightGBM', numeric_features + ['Sex'], n_repeats=5)
# Overall: 0.14804 LightGBM

# XGBoost with RMSE objective
# Hyperparameters were tuned with Optuna
xgb_params = {'grow_policy': 'lossguide', 'n_estimators': 300, 'learning_rate': 0.09471805900675286, 'max_depth': 8, 'reg_lambda': 33.33929116223339, 'min_child_weight': 27.048028004026204, 'colsample_bytree': 0.6105442825961575, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'gamma': 0, 'enable_categorical': True} # 0.14859
model = TransformedTargetRegressor(xgboost.XGBRegressor(**xgb_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'XGBoost', numeric_features + ['Sex'], n_repeats=5)
# Overall: 0.14853 XGBoost

# Catboost
# Hyperparameters were tuned with Optuna
cb_params = {'grow_policy': 'SymmetricTree', 'n_estimators': 1000, 'learning_rate': 0.128912681527133, 'l2_leaf_reg': 1.836927907521674, 'max_depth': 6, 'colsample_bylevel': 0.6775373040510968, 'random_strength': 0, 'boost_from_average': True, 'loss_function': 'RMSE', 'cat_features': ['Sex'], 'verbose': False} # 0.14847
model = TransformedTargetRegressor(catboost.CatBoostRegressor(**cb_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'Catboost', log_features + ['Sex'], n_repeats=5)
# Overall: 0.14851 Catboost

# # Hyperparameter tuning ==============================

# TODO: How to use optuna to tune hyperparameters for random forest?


def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
    max_depth = trial.suggest_int("max_depth", 5, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
    
    # Create the random forest regressor with the suggested hyperparameters
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features,
                                  random_state=seed)
    
    # Perform cross-validation
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    scores = []
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_log_error(y_val, y_pred))
        scores.append(score)
    
    # Return the mean score as the objective value
    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the random forest with the best hyperparameters
best_model = RandomForestRegressor(**best_params, random_state=seed)
best_model.fit(X, y)

# # Random forest
# def objective(trial):

## Random Forest

## LGBM 
# def objective(trial,params):
#     score = []
#     for i,(tr,val) in tqdm(enumerate(RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=seed).split(X,y)),total = 4):
#         X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]
#         lgbmmodel = LGBMRegressor(**lgbm_params)
#         lgbmmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)], eval_names=["valid"],eval_metric=['MSLE'])
#         msle = mean_squared_log_error(y_test, lgbmmodel.predict(X_test))
#         rmsle = np.sqrt(msle)
#         score.append(rmsle)
#     print(f" > RMSLE of LGBM =", score, file = sys.stderr)
#     return np.mean(score)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100,timeout=5000)


## XGB

## Catboost



# Other data:


# # Define a function for RMSLE
# def rmsle(y_true, y_pred):
#     return np.sqrt(mean_squared_log_error(y_true, y_pred))

# # Create a scorer for RMSLE
# rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# # Define the model pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lgbm', LGBMRegressor(objective='regression'))
# ])

# # Parameters grid to search
# param_grid = {
#     'lgbm__num_leaves': [31, 50, 100],
#     'lgbm__learning_rate': [0.05, 0.1],
#     'lgbm__n_estimators': [1500]
# }

# # Setup the grid search
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=rmsle_scorer, verbose=1)

# # Run grid search
# grid_search.fit(X_train, y_train)

# # Best parameters
# print("Best parameters found: ", grid_search.best_params_)

# # Best model
# best_model = grid_search.best_estimator_

# # Predict on the test set using the best model
# y_pred = best_model.predict(X_test)

# # Evaluate the best model using RMSLE
# rmsle_value = rmsle(y_test, y_pred)
# print(f'Root Mean Squared Logarithmic Error of the best model: {rmsle_value}')

# # - `Sex` is a categorical feature. For some models, we'll need to one-hot encode it, for other models it suffices to mark it as categorical.


