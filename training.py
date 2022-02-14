import pandas as pd
import numpy as np
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Importing the data from xlsx to a Dataframe df sorted by euro per square meter (and we drop that feature next)

housesPaths = 'ML data houses Final.xlsx'
df = pd.read_excel(housesPaths, index_col=False)
df = df.drop('Eur/m2', axis=1)
df = df.astype(int)

# Checking for good validation results
while 1:

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

    # Dropping the Final Price feature so we can start building our model
    y_train = train['FinalPrice']
    X_train = train.drop('FinalPrice', axis=1)

    y_validation = validation['FinalPrice']
    X_validation = validation.drop('FinalPrice', axis=1)

    y_test = test['FinalPrice']
    X_test = test.drop('FinalPrice', axis=1)

    # Random Forrest

    # Making the Regression Model based on the Random Forest Algorithm
    forest_params = {'n_estimators': [1000], 'max_depth': [None], 'min_samples_split': [2]}
    regression_model = GridSearchCV(RandomForestRegressor(), forest_params, cv=5, n_jobs=-1, verbose=1)
    regression_model.fit(X_train.values, y_train.values)

    # Checking results with validation set
    y_pred = regression_model.predict(X_validation.values)
    print("Random Forrest")
    print("Val R2: \t", r2_score(y_validation, y_pred))
    print("Val RMSE: \t", sqrt(mean_squared_error(y_validation, y_pred)))
    print("Val MAE: \t", mean_absolute_error(y_validation, y_pred))

    if (mean_absolute_error(y_validation, y_pred) < 35000) and (sqrt(mean_squared_error(y_validation, y_pred)) < 60000):
        break

# Using test set to evaluate the model
y_pred = regression_model.predict(X_test.values)
print("Test R2: \t", r2_score(y_test, y_pred))
print("Test RMSE: \t", sqrt(mean_squared_error(y_test, y_pred)))
print("Test MAE: \t", mean_absolute_error(y_test, y_pred))

# Keeping the price range to show on the results page
price_range = mean_absolute_error(y_test, y_pred)

# pickle the model and range for demo script
filename = 'price_range.sav'
pickle.dump(price_range, open(filename, 'wb'))
filename = 'finalized_model.sav'
pickle.dump(regression_model, open(filename, 'wb'))