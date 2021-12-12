import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model

#Read DataSet
df = pd.read_csv("Data/FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#plot independent var vs Dependent var
pyplot.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
pyplot.xlabel("Engine")
pyplot.ylabel("CO2 Emission")
pyplot.show()

pyplot.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
pyplot.xlabel("Cylinder")
pyplot.ylabel("CO2 Emission")
pyplot.show()

pyplot.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
pyplot.xlabel("Fuel Consumption")
pyplot.ylabel("CO2 Emission")
pyplot.show()

#Split Train-Test Data
mask = np.random.rand(len(cdf)) < 0.8
train= cdf[mask]
test= cdf[~mask]

#Model - Extension of Linear Regression
regr= linear_model.LinearRegression()
x  = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print("Coefficients " , regr.coef_)

#Predict
y_ = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x  = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y  = np.asanyarray(test[['CO2EMISSIONS']])

#Error
print("Residual sum square error : %.2f" %np.mean((y - y_) ** 2))
print("Variance score : %.2f" %regr.score(x,y))
      
