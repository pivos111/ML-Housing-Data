# How to run
To run the program you will need:
- Python 3.9 or newer
- .NET Core 6.0 or newer
- (If using MacOS or Linux Dist) Wine newest version

# How to use (.exe)
To run the app and get results:

- Step 1:
Run Ml UI.exe

- Step 2:
Fill the cells with data of the house you want to predict

- Step 3:
Press the 'Νέο ML Μοντέλο' button to create a new model
(step 2 and 3 can be done in any order)

- Step 4:
Press the 'Υπολογισμός' button to get the prediction

# How to use (Manually)
To run the scripts and get results:
(Get sure you have the required Libraries installed as seen in Extras/Requirements.txt)

- Step 1:
Run training.py

- Step 2:
Create a .txt file with a number in a new line for each of the following in order: ['SquareMeters', 'Rooms', 'Baths', 'Floor',
       'Parking', 'Apartement', 'Maisonette', 'Studio', 'SingleFamilyHouse',
       'Building', 'Complex', 'AutoElectricity', 'AutoOil', 'AutoGas',
       'FloorHeating', 'CentralOil', 'CentralGas', 'Age']
(Use only numbers, use 0 or 1 from Apartement to CentralGas, use number of years built for Age [ex. 15])

- Step 3:
Run demo.py

- Step 4:
The results are in PredictedPrice.txt.
The first line is the Predicted Price and the second line has the Price Range.

# Check the Extras Folder for the report etc.
