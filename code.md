```python
# Data preprocessing
import seaborn as sns, pandas as pd, numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from itertools import chain

filepath = 'data.csv'
data = pd.read_csv(filepath, sep=',')

calculator = Calculator(descriptors, ignore_3D=True)
calculator_descriptor = Calculator(descriptors, ignore_3D=True).descriptors


ReCiPe_Total = [x for x in data.columns if x in 'ReCiPe - Total']
SMILE = [x for x in data.columns if x in 'SMILE']
l_smile=data[SMILE].values.tolist()
A =list(chain.from_iterable(l_smile))


mols = [Chem.MolFromSmiles(smi) for smi in A]
res = [i for i in range(len(mols)) if mols[i] == None]
mols = [i for i in mols if i is not None]


df = calculator.pandas(mols)


df.isnull().sum().sum()


y_ReCiPe = data[ReCiPe_Total].drop(data[ReCiPe_Total].index[res])
y_ReCiPe = y_ReCiPe.reset_index(drop=True)


num_cols = df_prepared.select_dtypes('number').columns
obj_cols = df_prepared.select_dtypes('object').columns
bool_cols = df_prepared.select_dtypes('bool').columns


for c in df_prepared[obj_cols]:
    df_prepared[c] = pd.to_numeric(df_prepared[c], errors='coerce')
df_prepared[bool_cols] = df_prepared[bool_cols].replace({True: 1, False: 0})


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
empty_col = ['SpAbs_Dt','SpMax_Dt','SpDiam_Dt','SpAD_Dt','SpMAD_Dt','LogEE_Dt','SM1_Dt','VE1_Dt','VE2_Dt','VE3_Dt','VR1_Dt','VR2_Dt','VR3_Dt','DetourIndex']
df_delete_emp = df_prepared.drop(empty_col,axis=1)
df_delete_emp.to_csv('df_delete_emp.csv')


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform",add_indicator = False)
df_prepared_impute = pd.DataFrame(imputer.fit_transform(df_delete_emp), columns = df_delete_emp.columns)

sc = StandardScaler()
df_scaled = pd.DataFrame(sc.fit_transform(df_prepared_impute),columns = df_prepared_impute.columns)
X = sc.fit_transform(df_prepared_impute)
pca = PCA().fit(df_scaled)
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_prepared_impute.columns.values
loadings_df = loadings_df.set_index('variable')


import matplotlib.pyplot as plt
import numpy as np

components = np.arange(1, 213, step=1)
plt.ylim(0.0,1.1)
plt.plot(components, variance, marker='o', linestyle='--', color='green')
plt.xlabel('Number of PC')
plt.xticks(np.arange(0, 213, step=20))
plt.ylabel('Cumulative variance (%)')
plt.title('Cumulative explained variance')
plt.axhline(y=0.80, color='r', linestyle='-')
plt.text(40, 0.85, '80% variance threshold', color = 'red', fontsize=10)
plt.text(40, 0.70, "Components needed: 19", color = "red", fontsize=10)
plt.show()


for n in range(df_scaled.shape[1]):
    pca = PCA(n_components = n)
    PC = pca.fit(df_scaled)
    new_df = PC.transform(df_scaled)
    cum = sum(PC.explained_variance_ratio_)
    if cum>0.80:
        break
        

from pca import pca
model = pca()
out = model.fit_transform(df_scaled)
print(out['topfeat'])
selected_feature = out['topfeat'].feature[:19]
df_preprocessed = pd.DataFrame(data=new_df,columns=selected_feature)


sns.set(style="darkgrid")
sns.boxplot(data=y_ReCiPe)
plt.show()


import pandas as pd
import numpy as np
index=[]
q1 = y_ReCiPe.quantile(0.25)
q3 = y_ReCiPe.quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
#    df.at[i,0] == -1
for i in y_ReCiPe.index:
    if y_ReCiPe.at[i,'ReCiPe - Total'] > float(Upper_tail) or y_ReCiPe.at[i,'ReCiPe - Total'] < float(Lower_tail):
        index.append(i)

        
import pandas as pd
import numpy as np
out=[]
def iqr_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df.iloc[:,0]:
        if i > float(Upper_tail) or i < float(Lower_tail):
            out.append(i)
    print("Outliers:",out)        

df_preprocessed = df_preprocessed.drop(index)
y_ReCiPe = y_ReCiPe.drop(index)
y_ReCiPe = y_ReCiPe.reset_index(drop=True)
df_preprocessed = df_preprocessed.reset_index(drop=True)


# Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()
mm = MinMaxScaler()
y_ReCiPe_scaled = pd.DataFrame(mm.fit_transform(y_ReCiPe),columns=y_ReCiPe.columns)


from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb

X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(df_preprocessed, y_ReCiPe_scaled, test_size=0.10, random_state=93)

param = {'lambda':  [0.001, 0.01, 0.1],
        'alpha':  [0.001, 0.01, 0.1],
        'colsample_bytree': [0.2,0.4,0.6,0.8, 1.0],
        'subsample':  [0.4,0.6,0.8,1.0],
        'learning_rate': [0.001,0.01,0.015,0.02],
        'n_estimators': 10000,
        'max_depth': [5,7,9,11,13],
        'random_state':  [66],
        'min_child_weight':[1, 300]}

grid_search = GridSearchCV(estimator = regressor_R,
                           param_grid = param,
                           scoring = 'mean_squared_error',
                           cv = 5)
grid_search = grid_search.fit(X_train_R, y_train_R)


# XGBR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
regressor_R=xgb.XGBRegressor(eval_metric=mean_squared_error, max_depth=3, subsample=0.3)
regressor_R.fit(X_train_R, y_train_R, eval_set=[(X_test_R,y_test_R)])


predictions_R = regressor_R.predict(df_preprocessed)
predictions_test_R = regressor_R.predict(X_test_R)
predictions_R_train = regressor_R.predict(X_train_R)

MSE_R = mean_squared_error(y_test_R, predictions_test_R)
RMSE_R = np.sqrt( mean_squared_error(y_test_R, predictions_test_R) )
r2_R_test= r2_score(y_test_R, predictions_test_R)
MAE_R = mean_absolute_error(y_test_R, predictions_test_R)
EVC_R = explained_variance_score(y_ReCiPe_scaled, predictions_R)
r2_R_train = r2_score(y_train_R, predictions_R_train, multioutput='variance_weighted')
r2_R = r2_score(y_ReCiPe_scaled, predictions_R, multioutput='variance_weighted')
print("The MSE score is %.5f" % MSE_R )
print("The RMSE score is %.5f" % RMSE_R )
print("The MAE score is %.5f" % MAE_R )
print("The EVC score is %.5f" % EVC_R )
print("The R2 score of test data set is %.5f" % r2_R_test)
print("The R2 score of training dataset is %.5f" % r2_R_train )
print("The R2 score of the whole dataset is %.5f" % r2_R )

y_train_pred_R = regressor_R.predict(X_train_R)
ax = range(len(y_train_R))
plt.plot(ax, y_train_R, label="original")
plt.plot(ax, y_train_pred_R, label="predicted")
plt.title("ReCiPe Indicator regression")
plt.legend()
plt.show()

ax = range(len(y_test_R))
plt.plot(ax, y_test_R, label="original")
plt.plot(ax, predictions_test_R, label="predicted")
plt.title("ReciPe Indicator prediction")
plt.legend()
plt.show()

from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(regressor_R, max_num_features=8, ax=ax)
plt.show();


y_pred_R = regressor_R.predict(df_preprocessed)
fig, ax = plt.subplots(figsize = (9, 9))
ax.scatter(y_ReCiPe_scaled, y_pred_R, s=60, alpha=0.7, edgecolors="k")
xseq = np.linspace(0, 1, num=100)
ax.plot(xseq, xseq, color="k", lw=2.5)
plt.xlabel("Original ReCiPe indicator value")
plt.ylabel("Predicted ReCiPe indicator value");


y_test_pred_R = regressor_R.predict(X_test_R)
fig, ax = plt.subplots(figsize = (9, 9))
ax.scatter(y_test_R, y_test_pred_R, s=60, alpha=0.7, edgecolors="k")
xseq = np.linspace(0, 1, num=100)
ax.plot(xseq, xseq, color="k", lw=2.5)
plt.xlabel("Original ReCiPe indicator value")
plt.ylabel("Predicted ReCiPe indicator value");


# RFR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor as rfr
rf_model = rfr()
rf_model.fit(X_train_R, y_train_R)

predictions_R_test_rf = rf_model.predict(X_test_R)
predictions_R_train_rf = rf_model.predict(X_train_R)
predictions_R_rf = rf_model.predict(df_preprocessed)


MSE_R_rf = mean_squared_error(y_test_R, predictions_R_test_rf)
RMSE_R_rf = np.sqrt( mean_squared_error(y_test_R, predictions_R_test_rf) )
r2_R_test_rf = r2_score(y_test_R, predictions_R_test_rf, multioutput='variance_weighted')
MAE_R_rf = mean_absolute_error(y_test_R, predictions_R_test_rf)
EVC_R_rf = explained_variance_score(y_ReCiPe_scaled, predictions_R_rf)
r2_R_train_rf = r2_score(y_train_R, predictions_R_train_rf, multioutput='variance_weighted')
r2_R_rf = r2_score(y_ReCiPe_scaled, predictions_R_rf, multioutput='variance_weighted')
print("The MSE score is %.5f" % MSE_R_rf )
print("The RMSE score is %.5f" % RMSE_R_rf )
print("The MAE score is %.5f" % MAE_R_rf )
print("The EVC score is %.5f" % EVC_R_rf )
print("The R2 score of test data set is %.5f" % r2_R_test_rf)
print("The R2 score of training dataset is %.5f" % r2_R_train_rf )
print("The R2 score of the whole dataset is %.5f" % r2_R_rf )

y_train_pred_R = regressor_R.predict(X_train_R)
ax = range(len(y_train_R))
plt.plot(ax, y_train_R, label="original")
plt.plot(ax, y_train_pred_R, label="predicted")
plt.title("ReCiPe Indicator regression")
plt.legend()
plt.show()

ax = range(len(y_test_R))
plt.plot(ax, y_test_R, label="original")
plt.plot(ax, predictions_R_test_rf, label="predicted")
plt.title("ReciPe Indicator prediction")
plt.legend()
plt.show()

features = df_preprocessed.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[:8]
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(12,6))

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ANN
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
ann = Sequential()
ann.add(Dense(units=256,activation='relu',kernel_initializer='uniform'))
ann.add(Dropout(.4))
ann.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
ann.add(Dropout(.4))
ann.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
ann.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
ann.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=20,verbose=1)
model = ann.fit(X_train_R,y_train_R,validation_data=(X_test_R,y_test_R),epochs=200,batch_size=50,callbacks=[es])

fig, ax = plt.subplots()
ax.plot(model.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(model.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')

y_pred_test_R_ann = ann.predict(X_test_R)
y_train_R_ann = ann.predict(X_train_R)
y_R_ann = ann.predict(df_preprocessed)

MSE_R_ann = mean_squared_error(y_test_R, y_pred_test_R_ann)
RMSE_R_ann = np.sqrt( mean_squared_error(y_test_R, y_pred_test_R_ann) )
r2_R_ann_test = r2_score(y_test_R, y_pred_test_R_ann, multioutput='variance_weighted')
r2_R_ann_train = r2_score(y_train_R, y_train_R_ann, multioutput='variance_weighted')
r2_R_ann = r2_score(y_ReCiPe_scaled, y_R_ann, multioutput='variance_weighted')
MAE_R_ann = mean_absolute_error(y_test_R, y_pred_test_R_ann)
EVC_R_ann = explained_variance_score(y_ReCiPe_scaled, y_R_ann)
print("The RMSE score is %.5f" % RMSE_R_ann )
print("The MSE score is %.5f" % MSE_R_ann )
print("The MAE score is %.5f" % MAE_R_ann )
print("The EVC score is %.5f" % EVC_R_ann )
print("The R2 score is %.5f" % r2_R_ann_test)
print("The R2 score of training dataset is %.5f" % r2_R_ann_train)
print("The R2 score of the whole dataset is %.5f" % r2_R_ann )


y_train_pred_ann = ann.predict(X_train_R)
ax = range(len(y_train_pred_ann))
plt.plot(ax, y_train_R, label="original")
plt.plot(ax, y_train_pred_ann, label="predicted")
plt.title("ReCiPe Indicator regression")
plt.legend()
plt.show()

ax = range(len(y_test_R))
plt.plot(ax, y_test_R, label="original")
plt.plot(ax, y_pred_test_R_ann, label="predicted")
plt.title("ReciPe Indicator prediction")
plt.legend()
plt.show()

def cv_tune(optimizer,layer1,layer2,layer3):
    ann = Sequential()
    ann.add(Dense(units=layer1,activation='relu',kernel_initializer='uniform'))
    ann.add(Dense(units=layer2,activation='relu',kernel_initializer='uniform'))
    ann.add(Dense(units=layer3,activation='relu',kernel_initializer='uniform'))
    ann.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
    ann.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mean_squared_error'])
    return ann


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

ann = KerasRegressor(build_fn=cv_tune)
parameters = {'batch_size':[10,20,30],
             'epochs':[50, 100],
             'optimizer':['adam','rmsprop'],
             'units1':[512,256,128],
             'units2':[512,256,128],
             'units3':[512,256,128]}

grid_search = GridSearchCV(estimator = ann,
                           param_grid = parameters,
                           scoring = 'mean_squared_error',
                           cv = 5)
grid_search = grid_search.fit(X_train_R, y_train_R)
```

