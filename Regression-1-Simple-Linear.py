from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("Data\FuelConsumption.csv")
#print(df.describe())

cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
print(cdf.describe())
cdf.hist()
pyplot.show()

#Train - Test Split Data
msk = np.random.rand(len(df)) < 0.8  #uniformly distribute 0-1 over len(df) and compare with 0.8
train = cdf[msk]
test = cdf[~msk]
#print(len(train))
#print(len(test))

#plot engine vs emission using Train Data
pyplot.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
pyplot.xlabel('Engine Size')
pyplot.ylabel('CO2 Emissions')
#pyplot.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color = 'red')
pyplot.show()

#Model - Linear Regression
linearReg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
linearReg.fit(train_x,train_y)

#Coefficients
print(linearReg.intercept_)
print(linearReg.coef_)

#Plot Line
pyplot.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
pyplot.xlabel('Engine Size')
pyplot.ylabel('CO2 Emissions')
pyplot.plot(train_x, linearReg.coef_[0][0]*train_x + linearReg.intercept_[0], 'r')
pyplot.show()

#Test
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = linearReg.predict(test_x)

pyplot.plot(train_x, linearReg.coef_[0][0]*train_x + linearReg.intercept_[0], 'r')
pyplot.scatter(test_x, test_y, color='blue')
pyplot.scatter(test_x, test_y_, color='black')
pyplot.show()

#Evaluate Error
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
