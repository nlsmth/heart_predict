# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

import csv


# Set random seed
np.random.seed(0)

def runRandomForest(n_est, n_feat):
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    dataFrame['is_train'] = np.random.uniform(0, 1, len(dataFrame)) <= .7
    #print(dataFrame)
    #print(dataFrame.head())

    train, test = dataFrame[dataFrame['is_train']==True], dataFrame[dataFrame['is_train']==False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:',len(test))

    features = dataFrame.columns[:13]
    #print(features)

    clf = RandomForestClassifier(n_estimators=n_est, n_jobs=2, random_state=0, max_features=n_feat)

    #print(train['num'])
    clf.fit(train[features], train['num'])

    clf.predict(test[features])

    #print(clf.predict_proba(test[features])[0:10])
    predictions = clf.predict(test[features])

    results = pd.crosstab(test['num'], predictions, rownames=['Actual'], colnames=['Predicted'])
    print(results)

    #feature importance
    print(list(zip(train[features], clf.feature_importances_)))
    print(accuracy_score(test['num'],predictions))


def runKNN(k):
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    #print(dataFrame.head())
    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=k)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    # evaluate accuracy
    print(accuracy_score(y_test, pred))

def runSVM():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = LinearSVC(random_state=0)
    print(clf)
    clf.fit(X_train, y_train)
    #print(clf.coef_)
    #print(clf.intercept_)
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test,predictions))


#random 
#runRandomForest(500, 4)
#runKNN(15)
runSVM()