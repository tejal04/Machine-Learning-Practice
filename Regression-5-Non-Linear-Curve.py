import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

df = pd.read_csv("Data/ChinaGDP.csv")
#print(df.describe())
#print(df.head())

pyplot.figure(figsize=(8,5)) #width-height
x_data = df["Year"].values
y_data = df["Value"].values
pyplot.plot(x_data, y_data, 'ro')
#pyplot.plot(df.Year, df.Value, 'ro')
pyplot.ylabel('GDP')
pyplot.xlabel('Year')
pyplot.show()

#graph analysis - Logistic Function
#starts slow - increasing growth in middle - decreasing in end

#Model
''' beta1 controls curve's steepness
    beta2 slides the curve on x-axis'''
def sigmoid(x, beta1, beta2) :
    y = 1/ (1 + np.exp(-beta1 * (x-beta2)))
    return y

#verify model with plot
beta1 = 0.10
beta2 = 1990.0
y_pred = sigmoid(x_data, beta1, beta2)
pyplot.plot(x_data, y_data, 'ro')
pyplot.plot(x_data, y_pred*15000000000000)
pyplot.show()

#ToDo - Find best parameters for our model

#normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

'''
 use curve_fit which uses non-linear least squares
 to fit our sigmoid function, to data.
 Optimal values for the parameters so that
 the sum of the squared residuals of
 sigmoid(xdata, *popt) - ydata is minimized.
'''

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#final optimized model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
pyplot.figure(figsize=(8,5))
y = sigmoid(x, *popt)
pyplot.plot(xdata, ydata, 'ro', label='data')
pyplot.plot(x,y, linewidth=3.0, label='fit')
pyplot.legend(loc='best')
pyplot.ylabel('GDP')
pyplot.xlabel('Year')
pyplot.show()

# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
