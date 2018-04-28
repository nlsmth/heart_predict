# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

import csv

# Set random seed
np.random.seed(0)

data = []
with open('processedcleveland.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        missing = False
        for item in row:
            if item == "?":
                missing = True
        if not missing: 
            data.append(row)

#print(data)

for i in range(1, len(data)):
    if(float(data[i][13]) > 0):
        data[i][13] = 1
    for j in range(0, len(data[i])):
        data[i][j] = float(data[i][j])

dataFrame = pd.DataFrame(data[1:], columns=data[:1][0])


dataFrame['is_train'] = np.random.uniform(0, 1, len(dataFrame)) <= .7
#print(dataFrame)
#print(dataFrame.head())

train, test = dataFrame[dataFrame['is_train']==True], dataFrame[dataFrame['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

features = dataFrame.columns[:13]
#print(features)

clf = RandomForestClassifier(n_jobs=2, random_state=0)

#print(train['num'])
clf.fit(train[features], train['num'])

clf.predict(test[features])

#print(clf.predict_proba(test[features])[0:10])
predictions = clf.predict(test[features])

results = pd.crosstab(test['num'], predictions, rownames=['Actual'], colnames=['Predicted'])
print(results)

print(list(zip(train[features], clf.feature_importances_)))