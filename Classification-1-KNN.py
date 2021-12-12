import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing

df = pd.read_csv("Data/TeleCustomers.csv")
print(df.describe())
#df['custcat'].value_counts()
#df.columns

'''
convert pandas df to numpy array to use scikit library
'''
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
print(X[0:5])

y = df['custcat'].values
print(y[0:5])

#NORMALIZE Data - Data Stdardization gives zero mean, unit variance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Train KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#predict
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#Accuracy with different K
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

#Evaluation
pyplot.plot(range(1,Ks),mean_acc,'g')
pyplot.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
pyplot.legend(('Accuracy ', '+/- 3xstd'))
pyplot.ylabel('Accuracy ')
pyplot.xlabel('Number of Neighbours (K)')
pyplot.tight_layout()
pyplot.show()
