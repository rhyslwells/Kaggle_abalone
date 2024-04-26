import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# TODO: Add the following imports

# [ ] sdv
    


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

# LGBMRegressor
"""
Explain LGBMRegressor

"""

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

# Baseline Models 2 ==============================

# Random forest
model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
                                        remainder='passthrough'),
                      TransformedTargetRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=5),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1))
cross_validate(model, 'Random forest', log_features + ['Sex'])
# Overall: 0.14962 Random forest

# XGBoost with RMSE objective
# Hyperparameters were tuned with Optuna
xgb_params = {'grow_policy': 'lossguide', 'n_estimators': 300, 'learning_rate': 0.09471805900675286, 'max_depth': 8, 'reg_lambda': 33.33929116223339, 'min_child_weight': 27.048028004026204, 'colsample_bytree': 0.6105442825961575, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'gamma': 0, 'enable_categorical': True} # 0.14859
model = TransformedTargetRegressor(xgboost.XGBRegressor(**xgb_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'XGBoost', numeric_features + ['Sex'], n_repeats=5)
# Overall: 0.14853 XGBoost

# LightGBM
# Hyperparameters were tuned with Optuna
lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.038622511348472645, 'colsample_bytree': 0.5757189042456357, 'reg_lambda': 0.09664116733307193, 'min_child_samples': 87, 'num_leaves': 43, 'verbose': -1} # 0.14804
model = TransformedTargetRegressor(lightgbm.LGBMRegressor(**lgbm_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'LightGBM', numeric_features + ['Sex'], n_repeats=5)
# Overall: 0.14804 LightGBM

# Catboost
# Hyperparameters were tuned with Optuna
cb_params = {'grow_policy': 'SymmetricTree', 'n_estimators': 1000, 'learning_rate': 0.128912681527133, 'l2_leaf_reg': 1.836927907521674, 'max_depth': 6, 'colsample_bylevel': 0.6775373040510968, 'random_strength': 0, 'boost_from_average': True, 'loss_function': 'RMSE', 'cat_features': ['Sex'], 'verbose': False} # 0.14847
model = TransformedTargetRegressor(catboost.CatBoostRegressor(**cb_params),
                                                 func=np.log1p,
                                                 inverse_func=np.expm1)
cross_validate(model, 'Catboost', log_features + ['Sex'], n_repeats=5)
# Overall: 0.14851 Catboost

# # Hyperparameter tuning ==============================
#generalised function.
# # LGBM 
# def objective(trial):
#     lgbm_params = {
#         "random_state": seed,
#         'n_estimators' : 5000,        
#         "max_depth":trial.suggest_int('max_depth',5,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-3, 0.1, log=True),
#         "min_child_weight" : trial.suggest_float('min_child_weight', 0.5,4),
#         "min_child_samples" : trial.suggest_int('min_child_samples',1,250),
#         "subsample" : trial.suggest_float('subsample', 0.2, 1),
#         "subsample_freq" : trial.suggest_int('subsample_freq',0,5),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         'num_leaves' : trial.suggest_int('num_leaves', 8, 64),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         "metric": trial.suggest_categorical("metric", ["rmse","huber","quantile"]),
#         "boosting_type": "gbdt",    
#         "objective":'regression',
#         "device": "gpu",
#         "verbose": -1,
#         "early_stopping_rounds" : 1000
#     }
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

# # XGB 
# def objective(trial):
#     xgb_params = {
#         'n_estimators' : 5000,
#         'max_depth':  trial.suggest_int('max_depth',3,8),
#         "max_bin": trial.suggest_int('max_bin',128,512),
#         'subsample': trial.suggest_float('subsample', 0.2, 1),
#         'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
#         'gamma': trial.suggest_float("gamma", 1e-4, 1.0,log = True),
#         'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
#         'min_child_weight': trial.suggest_float('min_child_weight', 2,4),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-3, 0.2,log=True),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
#         "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1),
#         "grow_policy" : trial.suggest_categorical("grow_policy",["depthwise","lossguide"]),
#         "objective" : trial.suggest_categorical("objective",["reg:quantileerror","reg:squaredlogerror","reg:squarederror"]),
#         "tree_method" : "gpu_hist",
#         "early_stopping_rounds" : 1000,
#         "random_state" : seed,
#         "eval_metric": "rmsle",
#         "verbosity" :  0,
#     }
#     if xgb_params["objective"] == "reg:quantileerror":
#         xgb_params["quantile_alpha"] = trial.suggest_float('quantile_alpha', 0.1, 1.0, log=True)

#     score = []
#     for i,(tr,val) in tqdm(enumerate(RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=seed).split(X,y)),total = 4):
#         X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]

#         xgbmodel = XGBRegressor(**xgb_params)
#         xgbmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)],verbose=0,
#                      callbacks=[EarlyStopping(rounds = xgb_params["early_stopping_rounds"],save_best=True)])

#         msle = mean_squared_log_error(y_test, xgbmodel.predict(X_test))
#         rmsle = np.sqrt(msle)
#         score.append(rmsle)
#     print(f" > RMSLE of XGB =", score, file = sys.stderr)
#     return np.mean(score)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100,timeout=5000)

# # LGBM 
# def objective(trial):
#     lgbm_params = {
#         "random_state": seed,
#         'n_estimators' : 5000,        
#         "max_depth":trial.suggest_int('max_depth',5,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-3, 0.1, log=True),
#         "min_child_weight" : trial.suggest_float('min_child_weight', 0.5,4),
#         "min_child_samples" : trial.suggest_int('min_child_samples',1,250),
#         "subsample" : trial.suggest_float('subsample', 0.2, 1),
#         "subsample_freq" : trial.suggest_int('subsample_freq',0,5),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         'num_leaves' : trial.suggest_int('num_leaves', 8, 64),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         "metric": trial.suggest_categorical("metric", ["rmse","huber","quantile"]),
#         "boosting_type": "gbdt",    
#         "objective":'regression',
#         "device": "gpu",
#         "verbose": -1,
#         "early_stopping_rounds" : 1000
#     }
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


# # XGB 
# def objective(trial):
#     xgb_params = {
#         'n_estimators' : 5000,
#         'max_depth':  trial.suggest_int('max_depth',3,8),
#         "max_bin": trial.suggest_int('max_bin',128,512),
#         'subsample': trial.suggest_float('subsample', 0.2, 1),
#         'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
#         'gamma': trial.suggest_float("gamma", 1e-4, 1.0,log = True),
#         'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
#         'min_child_weight': trial.suggest_float('min_child_weight', 2,4),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-3, 0.2,log=True),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
#         "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1),
#         "grow_policy" : trial.suggest_categorical("grow_policy",["depthwise","lossguide"]),
#         "objective" : trial.suggest_categorical("objective",["reg:quantileerror","reg:squaredlogerror","reg:squarederror"]),
#         "tree_method" : "gpu_hist",
#         "early_stopping_rounds" : 1000,
#         "random_state" : seed,
#         "eval_metric": "rmsle",
#         "verbosity" :  0,
#     }
#     if xgb_params["objective"] == "reg:quantileerror":
#         xgb_params["quantile_alpha"] = trial.suggest_float('quantile_alpha', 0.1, 1.0, log=True)

#     score = []
#     for i,(tr,val) in tqdm(enumerate(RepeatedStratifiedKFold(n_splits=4, n_repeats=1,random_state=seed).split(X,y)),total = 4):
#         X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]

#         xgbmodel = XGBRegressor(**xgb_params)
#         xgbmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)],verbose=0,
#                      callbacks=[EarlyStopping(rounds = xgb_params["early_stopping_rounds"],save_best=True)])

#         msle = mean_squared_log_error(y_test, xgbmodel.predict(X_test))
#         rmsle = np.sqrt(msle)
#         score.append(rmsle)
#     print(f" > RMSLE of XGB =", score, file = sys.stderr)
#     return np.mean(score)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100,timeout=5000)


test = pd.read_csv("/kaggle/input/playground-series-s4e4/test.csv",index_col="id")
test["Sex"]  = le.transform(test["Sex"])
test["Volume"] = test["Length"]*test["Diameter"]*test["Height"] 
# test["Density"] = test["Whole weight"]/test["Volume"]
test['Weight Remaining'] = test['Whole weight'] - test['Whole weight.1'] - test['Whole weight.2']
test["log_Volume"] = np.log(test["Volume"]+1)


submission = pd.DataFrame()
submission["id"] = test.index
submission["Rings"] = 0

xgb_score = []
xgb_params = {'max_depth': 7, 
              'max_bin': 343, 
              'subsample': 0.7790858322788681, 
              'alpha': 0.20656094847473275, 
              'gamma': 0.00013022972426240075, 
              'lambda': 0.003441137117852323, 
              'min_child_weight': 2.0063991207235343, 
              'learning_rate': 0.012558395627618084, 
              'colsample_bytree': 0.9760133525508788, 
              'colsample_bylevel': 0.7770210954947634, 
              'colsample_bynode': 0.8820455137287587, 
              'grow_policy': 'lossguide', 
              'objective': 'reg:squaredlogerror'}

xgbmodel = XGBRegressor(random_state=seed)
xgbmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)],verbose = 0,callbacks=[EarlyStopping(rounds = 4000,save_best=True)])

msle = mean_squared_log_error(y_test, xgbmodel.predict(X_test))
rmsle = np.sqrt(msle)
xgb_score.append(rmsle)
submission["Rings"] += xgbmodel.predict(test)
submission.to_csv("results.csv",header=True,index=False)
submission[["id","Rings"]].to_csv("submission.csv",header=True,index=False)


# Other data:


# # Feature engineering ==============================

# # Basic calculations see eda notebooks

# # Model building ==============================
# from sklearn.metrics import mean_squared_log_error, make_scorer

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



# # Model training ==============================
# X = train.drop(["Rings"],axis=1)
# y = train["Rings"]



# # basic models before hyperparameter tuning

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# def cross_validate(model, label, features=test.columns, n_repeats=1):
#     """
    
    
#     # Cross-validation

#     To ensure that our cross-validation results are consistent, we'll use the same function for cross-validating all models.

#     Notice that in cross-validation, we first split the dataset and then add the original data only 
#     to the training dataset. The validation dataset consists purely of competition data. 
#     This setup lets us correctly assess whether the original data are useful or harmful.
    

#     Compute out-of-fold and test predictions for a given model.
    
#     Out-of-fold and test predictions are stored in the global variables
#     oof and test_pred, respectively.
    
#     If n_repeats > 1, the model is trained several times with different seeds.
    
#     All predictions are clipped to the interval [1, 29].
#     """
#     scores = []
#     oof_preds = np.full_like(train.Rings, np.nan, dtype=float)
#     for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.Rings)):
#         X_tr = train.iloc[idx_tr][features]
#         X_va = train.iloc[idx_va][features]
#         y_tr = train.iloc[idx_tr].Rings
#         y_va = train.iloc[idx_va].Rings
        
#         if USE_ORIGINAL_DATA:
#             X_tr = pd.concat([X_tr, original_dataset[features]], axis=0)
#             y_tr = pd.concat([y_tr, original_dataset.Rings], axis=0)
            
#         y_pred = np.zeros_like(y_va, dtype=float)
#         for i in range(n_repeats):
#             m = clone(model)
#             if n_repeats > 1:
#                 mm = m
#                 if isinstance(mm, Pipeline):
#                     mm = mm[-1]
#                 if isinstance(mm, TransformedTargetRegressor):
#                     mm = mm.regressor
#                 mm.set_params(random_state=i)
#             m.fit(X_tr, y_tr)
#             y_pred += m.predict(X_va)
#         y_pred /= n_repeats
#         y_pred = y_pred.clip(1, 29)
        
# #         residuals = np.log1p(y_va) - np.log1p(y_pred)
# #         plt.figure(figsize=(6, 2))
# #         plt.scatter(y_pred, residuals, s=1)
# #         plt.axhline(0, color='k')
# #         plt.show()
        
#         score = mean_squared_log_error(y_va, y_pred, squared=False)
#         print(f"# Fold {fold}: RMSLE={score:.5f}")
#         scores.append(score)
#         oof_preds[idx_va] = y_pred
#     print(f"{Fore.GREEN}# Overall: {np.array(scores).mean():.5f} {label}{Style.RESET_ALL}")
#     oof[label] = oof_preds
    
#     if COMPUTE_TEST_PRED:
#         # Retrain n_repeats times with the whole dataset and average
#         y_pred = np.zeros(len(test), dtype=float)
#         X_tr = train[features]
#         y_tr = train.Rings
#         if USE_ORIGINAL_DATA:
#             X_tr = pd.concat([X_tr, original_dataset[features]], axis=0)
#             y_tr = pd.concat([y_tr, original_dataset.Rings], axis=0)
#         for i in range(n_repeats):
#             m = clone(model)
#             if n_repeats > 1:
#                 mm = m
#                 if isinstance(mm, Pipeline):
#                     mm = mm[-1]
#                 if isinstance(mm, TransformedTargetRegressor):
#                     mm = mm.regressor
#                 mm.set_params(random_state=i)
#             m.fit(X_tr, y_tr)
#             y_pred += m.predict(test[features])
#         y_pred /= n_repeats
#         y_pred = y_pred.clip(1, 29)
#         test_pred[label] = y_pred


# # PolynomialFeatures + Ridge (linear model) one-hot-encode categroical
# model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
#                                         remainder='passthrough'),
#                       StandardScaler(),
#                       PolynomialFeatures(degree=3),
#                       TransformedTargetRegressor(Ridge(100),
#                                                  func=np.log1p,
#                                                  inverse_func=np.expm1))
# cross_validate(model, 'Poly-Ridge', numeric_features + log_features + ['Sex'])
# # Overall: 0.15293 Poly-Ridge

# # Random forest
# model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), ['Sex'])],
#                                         remainder='passthrough'),
#                       TransformedTargetRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=5),
#                                                  func=np.log1p,
#                                                  inverse_func=np.expm1))
# cross_validate(model, 'Random forest', log_features + ['Sex'])
# # Overall: 0.14962 Random forest

# lgbmmodel = LGBMRegressor(random_state=seed, verbose=-1)

# # xgbmodel = XGBRegressor(random_state=seed)
# # catmodel = CatBoostRegressor(random_state=seed, verbose=0)

# print("CV RMSLE score of LGBM is ",np.sqrt(-cross_val_score(lgbmmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))
# # print("CV RMSLE score of XGB is ",np.sqrt(-cross_val_score(xgbmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))
# # print("CV RMSLE score of CAT is ",np.sqrt(-cross_val_score(catmodel,X,y,cv=4, scoring = 'neg_mean_squared_log_error').mean()))

# # Refining hyperparameters using optuna

# lgbm_params = {
#     'n_estimators' : 15000,  
#     "random_state": seed,
#     "boosting_type": "gbdt",    
#     "objective":'regression',
#     "device": "gpu",
#     "verbose": -1,
#     "early_stopping_rounds" : 4000,
#     'max_depth': 9,
#     'learning_rate': 0.0754689136929529,
#     'min_child_weight': 2.9774820924588674,
#     'min_child_samples': 172,
#     'subsample': 0.749283862376052,
#     'subsample_freq': 0,
#     'colsample_bytree': 0.5668465666039963,
#     'num_leaves': 18,
#     'lambda_l1': 4.011146777594568e-05,
#     'lambda_l2': 0.18342984449081373,
#     'metric': 'huber'
# }
# SPLITS = 5
# REPEATS = 1
# lgbm_score = []

# #use RepeatedStratifiedKFold

# for i,(tr,val) in enumerate(RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS,random_state=seed).split(X,y)):
    
#     print("-"*30,f"FOLD {i+1}/{SPLITS*REPEATS}","-"*30)
#     X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]
    
#     print("\n->","LGBM:")
#     lgbmmodel = LGBMRegressor(**lgbm_params)
#     lgbmmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)], eval_names=["valid"],eval_metric=['MSLE'])
#     msle = mean_squared_log_error(y_test, lgbmmodel.predict(X_test))
#     rmsle = np.sqrt(msle)
#     lgbm_score.append(rmsle)
#     print(f"Fold {i+1} RMSLE of LGBM =", rmsle,"\n")
#     submission["Rings"] += lgbm_wt*lgbmmodel.predict(test)
    

# # With the same data sets compare models trained againest each other.

# # Submission