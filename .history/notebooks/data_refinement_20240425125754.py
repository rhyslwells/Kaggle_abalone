
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