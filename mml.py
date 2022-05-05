import pandas as pd
from sklearn.linear_model import LinearRegression

import pickle

df=pd.read_csv('FuelConsumption.csv')

#lets use required features
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Lets train the data and predictor variable
x=cdf.iloc[:,:3]
y=cdf.iloc[:,-1]

regressor=LinearRegression()
#Lets train the model
#Training the model with training data
regressor.fit(x,y)

#Saving the model to the current directory 
#Pickle serializes objects so they can be saved to a file, and loaded in a 
#program again later on.

pickle.dump(regressor,open('model.pkl','wb'))
