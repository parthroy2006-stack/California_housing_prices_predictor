from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


#Load the data
housing = pd.read_csv("housing.csv")

#Create a stratified test set
housing["income_cat"]= pd.cut(housing["median_income"],
                       bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                       labels=[1,2,3,4,5])

split= StratifiedShuffleSplit(n_splits =1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1)
    
#We will work o0n copy of training data
housing = strat_train_set.copy()

#Separate fetaures and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis=1)

print(housing,housing_labels)

#Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs =["ocean_proximity"]

# Lets make the pipeline for numerical attributes
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('scaler',StandardScaler())
])

# Lets make the pipeline for categorical attributes
cat_pipeline = Pipeline([
    ('onehot',OneHotEncoder(handle_unknown="ignore"))
])

#Construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

#Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

#Train the model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels,lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
#print("Linear Regression RMSE:",lin_rmse)
print(pd.Series(lin_rmses).describe())


#Decision Tree Regressor
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels,dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
#print("Decision Tree Regressor RMSE:",dec_rmse)  
print(pd.Series(dec_rmses).describe())

#Random Forest Regressor
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared,housing_labels)
rf_preds = rf_reg.predict(housing_prepared) 
#rf_rmse = root_mean_squared_error(housing_labels,rf_preds)
rf_rmses = -cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv=10)
#print(pd.Series(rf_rmses).describe())  
print(pd.Series(rf_rmses).describe())