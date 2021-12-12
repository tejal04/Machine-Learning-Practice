import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

#Read DataSet
df = pd.read_csv("Data/FuelConsumption.csv")
cdf = df[['ENGINESIZE','CO2EMISSIONS']]

#Split Train - Test Data
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test  = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#Model Data
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

#plot
pyplot.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
pyplot.plot(XX, yy, '-r' )
pyplot.xlabel("Engine size")
pyplot.ylabel("Emission")
pyplot.show()

#Test/Predict
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

#Evaluate Error
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

