import numpy as np
from matplotlib import pyplot

#line function 
#y = m(x) + c
x = np.arange(-5, 5, 0.1)
y_ = 2*x + 3
y_noise = 2*np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()

#quadratic function (parabola)
#y = ax^2 + bx + c
x = np.arange(-5, 5, 0.1)
y_ = x**2 + 3*x + 12
#y_ = np.power(x,2) + 3*x + 12
y_noise = 5*np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()

#cubic function
#y = ax^3 + bx^2 + cx + d
x = np.arange(-5, 5, 0.1)
y_ = x**3 + x**2 + x + 6
y_noise = 22*np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()

#exponential function
#y = a+bc^x
x = np.arange(-5, 5, 0.1)
y_ = np.exp(x) + 5
y_noise = 10*np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()

#logarithmic function
#y = a+blog(x)
x = np.arange(-5, 5, 0.1)
y_ = np.log(x)
y_noise =0.7* np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()

#Sigmoidal/Logistic
# a + [b / 1+(c^{x-d})]
x = np.arange(-5, 5, 0.1)
y_ = 1-4/(1+ np.power(3,x-2))
y_noise = np.random.normal(size=x.size)
y = y_ + y_noise
pyplot.plot(x, y, 'bo')
pyplot.plot(x, y_, 'red')
pyplot.show()
