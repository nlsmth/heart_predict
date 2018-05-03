from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import pandas as pd
import numpy as np
import csv

np.random.seed(0)

#START: Noel Smith
def runRandomForest():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [3, 5, 10, 20, 50, 75, 100, None],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 4, 6, 10],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [100, 250, 500, 1000, 2000]
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    dataFrame['is_train'] = np.random.uniform(0, 1, len(dataFrame)) <= .7

    train, test = dataFrame[dataFrame['is_train']==True], dataFrame[dataFrame['is_train']==False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:',len(test))

    features = dataFrame.columns[:13]

    clf = RandomForestClassifier()



    def grid_search_wrapper(refit_score='precision_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = RandomizedSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=2, n_iter=500)
                            
        grid_search.fit(dataFrame[features], dataFrame['num'])

        fin = pd.DataFrame(grid_search.cv_results_)
        fin = fin.sort_values(by='mean_test_precision_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_bootstrap', 'param_min_samples_leaf']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_recall_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_bootstrap', 'param_min_samples_leaf']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_accuracy_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_bootstrap', 'param_min_samples_leaf']].round(3).head(1))

        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')


def runKNN():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'n_neighbors' : np.arange(1, 25, 1)
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    clf = KNeighborsClassifier()

    def grid_search_wrapper(refit_score='precision_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)
        grid_search.fit(X, y)

        fin = pd.DataFrame(grid_search.cv_results_)
        fin = fin.sort_values(by='mean_test_' + refit_score , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_neighbors']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_recall_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_neighbors']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_accuracy_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_neighbors']].round(3).head(1))
        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')
#END: Noel Smith


#START: Michael Janvier
def runSVM():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'C' : [0.001, 0.01, 0.1, 1, 10],
        'gamma':[1e-1, 1, 1e1]
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1
    
    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    clf = svm.SVC(kernel = 'linear', probability=True)
    

    def grid_search_wrapper(refit_score='accuracy_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=2)
        grid_search.fit(X, y)

        fin = pd.DataFrame(grid_search.cv_results_)
        fin = fin.sort_values(by='mean_test_precision_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_C', 'param_gamma']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_recall_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_C', 'param_gamma']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_accuracy_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_C', 'param_gamma']].round(3).head(1))

        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')

def runAdaBoost():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'n_estimators' : [50, 100, 250, 500, 1000, 2000],
        'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3, .5, 1]
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1
    
    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    clf = AdaBoostClassifier()

    def grid_search_wrapper(refit_score='accuracy_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=2)
        grid_search.fit(X, y)
        
        fin = pd.DataFrame(grid_search.cv_results_)
        fin = fin.sort_values(by='mean_test_precision_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_estimators', 'param_learning_rate']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_recall_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_estimators', 'param_learning_rate']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_accuracy_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_estimators', 'param_learning_rate']].round(3).head(1))

        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')
#END: Michael Janvier


#START: Noel Smith
def runGradientBoost():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    
    
    param_grid = {
        'max_depth': [3, 5, 10, 20, 50, 75, 100, None],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 4, 6, 10],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [100, 250, 500, 1000, 2000],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, .5, 1]

    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1
    
    X = np.array(dataFrame.ix[:, 0:13]) 	# end index is exclusive
    y = np.array(dataFrame['num']) 	# another way of indexing a pandas df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = GradientBoostingClassifier(random_state=0)

    def grid_search_wrapper(refit_score='accuracy_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = RandomizedSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=2, n_iter=500)
        grid_search.fit(X, y)

        fin = pd.DataFrame(grid_search.cv_results_)
        fin = fin.sort_values(by='mean_test_precision_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_min_samples_leaf', 'param_learning_rate']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_recall_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_min_samples_leaf', 'param_learning_rate']].round(3).head(1))
        fin = fin.sort_values(by='mean_test_accuracy_score' , ascending=False)
        print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators', 'param_min_samples_leaf', 'param_learning_rate']].round(3).head(1))

        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')
#END: Noel Smith

#START: Michael Janvier
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def thresholds(y_test, y_pred):
    aScore = None
    leastFN = np.inf
    lowT = 0
    for t in np.arange(1.00, 0, -.01):
        FN = 0
        y_adjusted = adjusted_classes(y_pred, t)
        for i in range(len(y_adjusted)):
            if y_adjusted[i]==0 and y_test[i] != y_adjusted[i]:
                FN += 1
        if(FN < leastFN):
            aScore = accuracy_score(y_test, y_adjusted)
            leastFN = FN
            lowT = t
    print(lowT)
    print(leastFN)
    print(aScore)
    print('\n')


    #for a in aScores:
    #    print(a)

    


def getThresholds():

    dataFrame = pd.read_csv('processedclevelandPrime.csv')
    for i in range(0, len(dataFrame['num'])):
        if(dataFrame['num'][i] > 0):
            dataFrame['num'][i] = 1

    X = np.array(dataFrame.ix[:, 0:13])
    y = np.array(dataFrame['num'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    knnClassifierPrecision = KNeighborsClassifier(n_jobs=2, n_neighbors=4)
    knnClassifierPrecision.fit(X_train, y_train)
    y_scoresknnClassifierPrecision = knnClassifierPrecision.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresknnClassifierPrecision)

    knnClassifierRecall = KNeighborsClassifier(n_jobs=2, n_neighbors=13)
    knnClassifierRecall.fit(X_train, y_train)
    y_scoresknnClassifierRecall = knnClassifierRecall.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresknnClassifierRecall)

    knnClassifierAccuracy = KNeighborsClassifier(n_jobs=2, n_neighbors=23)
    knnClassifierAccuracy.fit(X_train, y_train)
    y_scoresknnClassifierAccuracy = knnClassifierAccuracy.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresknnClassifierAccuracy)




    adaBoostClassifierPrecision = AdaBoostClassifier(n_estimators=1000 , learning_rate=.001)
    adaBoostClassifierPrecision.fit(X_train, y_train)
    y_scoresadaBoostClassifierPrecision = adaBoostClassifierPrecision.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresadaBoostClassifierPrecision)

    adaBoostClassifierRecall = AdaBoostClassifier(n_estimators=250 , learning_rate=.01)
    adaBoostClassifierRecall.fit(X_train, y_train)
    y_scoresadaBoostClassifierRecall = adaBoostClassifierRecall.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresadaBoostClassifierRecall)

    adaBoostClassifierAccuracy = AdaBoostClassifier(n_estimators=250 , learning_rate=.01)
    adaBoostClassifierAccuracy.fit(X_train, y_train)
    y_scoresadaBoostClassifierAccuracy = adaBoostClassifierAccuracy.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresadaBoostClassifierAccuracy)




    rfClassifierPrecision = RandomForestClassifier(n_jobs=2, bootstrap=True , max_depth=3, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=1000)
    rfClassifierPrecision.fit(X_train, y_train)
    y_scoresrfClassifierPrecision = rfClassifierPrecision.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresrfClassifierPrecision)

    rfClassifierRecall = RandomForestClassifier(n_jobs=2, bootstrap=False , max_depth=50, max_features='sqrt', min_samples_leaf=4, min_samples_split=5, n_estimators=100)
    rfClassifierRecall.fit(X_train, y_train)
    y_scoresrfClassifierRecall = rfClassifierRecall.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresrfClassifierRecall)

    rfClassifierAccuracy = RandomForestClassifier(n_jobs=2, bootstrap=True , max_depth=3, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=1000)
    rfClassifierAccuracy.fit(X_train, y_train)
    y_scoresrfrfClassifierAccuracy = rfClassifierAccuracy.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresrfrfClassifierAccuracy)





    gbrfClassifierPrecision = GradientBoostingClassifier(learning_rate=.0001 , max_depth=50, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=100)
    gbrfClassifierPrecision.fit(X_train, y_train)
    y_scoresrfgbrfClassifierPrecision = gbrfClassifierPrecision.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresrfgbrfClassifierPrecision)

    gbrfClassifierRecall = GradientBoostingClassifier(learning_rate=1 , max_depth=5, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=2000)
    gbrfClassifierRecall.fit(X_train, y_train)
    y_scoresgbrfClassifierRecall = gbrfClassifierRecall.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresgbrfClassifierRecall)

    gbrfClassifierAccuracy = GradientBoostingClassifier(learning_rate=.001 , max_depth=3, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=500)
    gbrfClassifierAccuracy.fit(X_train, y_train)
    y_scoresgbrfClassifierAccuracy = gbrfClassifierRecall.predict_proba(X_test)[:, 1]
    thresholds(y_test, y_scoresgbrfClassifierAccuracy)

#END: Michael Janvier



#runRandomForest()
#runKNN()
#runSVM()
#runAdaBoost()
#runGradientBoost()

getThresholds()

