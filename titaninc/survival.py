import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv('train.csv')
y_train = X_train.pop('Survived').values

def preprocessingdata(X):
    X.pop('Name')
    X.pop('Cabin')
    X.pop('Ticket')
    X.pop('PassengerId')
    X = X.values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    X[:,2:3] = imputer.fit_transform(X[:,2].reshape(-1,1))
    imputer2 = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    X[:,6:] = imputer2.fit_transform(X[:,6].reshape(-1,1))
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
    X = ct.fit_transform(X)
    X = X[:,1:]
    ct2 = ColumnTransformer([('encoder',OneHotEncoder(),[6])],remainder='passthrough')
    X = ct2.fit_transform(X)
    X = X[:,1:]
    return X

X_train = preprocessingdata(X_train)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#building ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train,y_train, batch_size = 25, epochs = 100)

X_test = pd.read_csv("test.csv")

X_test = preprocessingdata(X_test)
X_test = sc_X.transform(X_test)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

resultnew=[]
for i in range(len(y_pred)):
    if y_pred[i][0] == False:
        result.append('n')
    else:
        result.append('y')
        
resultnew = pd.DataFrame(result, columns=['Output'])



