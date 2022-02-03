import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Importing the data from xlsx to a Dataframe df sorted by a variable

housesPaths = 'ML data houses only numbers.xlsx'
df = pd.read_excel(housesPaths)

# Making a train, validation and test dataframe for splitting the data

train = pd.DataFrame(columns=df.columns)
validation = pd.DataFrame(columns=df.columns)
test = pd.DataFrame(columns=df.columns)

v1 = t1 = t2 = 0                                                 #Initializing 3 flags

bins = round(df.shape[0]/10)                                    #Counting the number of 10s the data can be split into

for index,row in df.iterrows():
    if (index+1)/10 <= bins:                                    #If we have more than 10 data left
        if (index+1)%10==0 and index!=0:                        #If the index is at the last of 10 items
            
            # We check 3 out of 10 items from each set of 10 to be sent at the test
            # and validation Dataframe respectively (2 to test and 1 to validation)

            v1 = np.random.randint(0,10)                        # Random number inside the 0 - 9 bounds
            validation = validation.append(df.iloc[index-v1])   # The index is at the last of 10 items so we subtract the flag

            t1 = np.random.randint(1,10)                        # Same as validation set but we keep t1>t2 to keep DataFrame sorted
            while t1 == v1:                                     # Check for same item of set
                t1 = np.random.randint(1,10)
            test = test.append(df.iloc[index-t1])

            t2 = np.random.randint(0,t1)                        # Same for the second test item and we keep t1>t2
            while t2 == v1 or t2 == t1 :
                t2 = np.random.randint(0,10)
            test = test.append(df.iloc[index-t2])

            for i in range (9, -1, -1):                         # We insert the remaining 7 items in the train set
                if i!= t1 and i!= t2 and i!= v1:                # Checking for previously inserted items
                    train = train.append(df.iloc[index-i])
    else:                                                       # If there are less than 10 data left
        train = train.append(df.iloc[index])                    # insert them in the train dataset

# Normalazing Teliki Timi Feature

# x = df['Teliki Timi']
# sns.set_style('whitegrid')
# sns.displot(x)
# plt.show()

# df['Teliki Timi_log'] = np.log(df['Teliki Timi'])
# x = df['Teliki Timi_log']
# sns.displot(x)
# plt.show()

#df.style.background_gradient(cmap='Blues')

# data = df.corr()
# sns.heatmap(data.corr(), annot=True)
# plt.show()

# x = df['€/m^2']
# y = df['Tetragonika']
# plt.scatter(x, y)
# plt.show()

y_train = train['Teliki Timi']
X_train = train.drop('Teliki Timi', axis=1)

y_test = test['Teliki Timi']
X_test = test.drop('Teliki Timi', axis=1)

# Linear

## def train_eval(algorithm, grid_params, X_train, X_test, y_train, y_test):

# regression_model = GridSearchCV(LinearRegression(), {}, cv=5, n_jobs=-1, verbose=1)
# regression_model.fit(X_train, y_train)
# y_pred = regression_model.predict(X_test)
# print("Linear Regression")
# print("R2: \t", r2_score(y_test, y_pred))
# print("RMSE: \t", sqrt(mean_squared_error(y_test, y_pred)))
# print("MAE: \t", mean_absolute_error(y_test, y_pred))

## train_eval(LinearRegression(), {}, *housing_split)

# KNN

# knn_params = {'n_neighbors' : [1, 5, 10, 20, 30, 50, 75, 100]}
# regression_model = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, n_jobs=-1, verbose=1)
# regression_model.fit(X_train, y_train)
# y_pred = regression_model.predict(X_test)
# print("KNN")
# print("R2: \t", r2_score(y_test, y_pred))
# print("RMSE: \t", sqrt(mean_squared_error(y_test, y_pred)))
# print("MAE: \t", mean_absolute_error(y_test, y_pred))

## model = train_eval(KNeighborsRegressor(), knn_params, *housing_split)

# Decision Tree

# tree_params = {}
# regression_model = GridSearchCV(DecisionTreeRegressor(), tree_params, cv=5, n_jobs=-1, verbose=1)
# regression_model.fit(X_train, y_train)
# y_pred = regression_model.predict(X_test)
# print("Decision Tree")
# print("R2: \t", r2_score(y_test, y_pred))
# print("RMSE: \t", sqrt(mean_squared_error(y_test, y_pred)))
# print("MAE: \t", mean_absolute_error(y_test, y_pred))

# Random Forrest

forest_params = {'n_estimators': [1000], 'max_depth': [None], 'min_samples_split': [2]}
regression_model = GridSearchCV(RandomForestRegressor(), forest_params, cv=5, n_jobs=-1, verbose=1)
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)
print("Random Forrest")
print("R2: \t", r2_score(y_test, y_pred))
print("RMSE: \t", sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: \t", mean_absolute_error(y_test, y_pred))

x1 = y_test
x2 = y_pred
x3 = abs(x2 - x1)
y = test['Tetragonika']
# plt.scatter(x1, y, label = "line 1")
# plt.scatter(x2, y, label = "line 2")
plt.scatter(x3, y, label = "line 3")
plt.show()

# print(len(y_pred))
# print(y_test)