import pickle
import pandas as pd

# Load data from text file to be predicted
userlist = []
with open('UserValues.txt') as f:
    for line in f:
        userlist.append(int(line))

# Make a dataframe from the data
user_value = pd.DataFrame(list(userlist))
user_value.index= ['SquareMeters', 'Rooms', 'Baths', 'Floor',
       'Parking', 'Apartement', 'Maisonette', 'Studio', 'SingleFamilyHouse',
       'Building', 'Complex', 'AutoElectricity', 'AutoOil', 'AutoGas',
       'CentralGas', 'CentralOil', 'FloorHeating', 'Age']

# Load model and price range
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
range = pickle.load(open('price_range.sav', 'rb'))

# Use model to predict the value of the house
predicted_value = loaded_model.predict(user_value.values.reshape(1,-1))

# Transform values into strings
range = str(int(range))
predicted_value = str(predicted_value)
temp = predicted_value.split('[')[1]
pv = temp.split('.')[0]

# Pass the values into a txt
lines = [ pv ,range]
with open('PredictedPrice.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
