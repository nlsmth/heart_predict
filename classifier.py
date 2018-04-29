# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

import csv


# Set random seed
np.random.seed(0)
plt.style.use("ggplot")

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def runRandomForest():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'min_samples_split': [3, 5, 10], 
        'n_estimators' : [100, 300],
        'max_depth': [3, 5, 15, 25],
        'max_features': [3, 5, 10, 13]
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
    #print(dataFrame)
    #print(dataFrame.head())

    train, test = dataFrame[dataFrame['is_train']==True], dataFrame[dataFrame['is_train']==False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:',len(test))

    features = dataFrame.columns[:13]

    #clf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=0, max_features=n_feat)
    clf = RandomForestClassifier(n_jobs=-1)
    #clf.fit(train[features], train['num'])
    #clf.predict(test[features])

    #predictions = clf.predict(test[features])
    #results = pd.crosstab(test['num'], predictions, rownames=['Actual'], colnames=['Predicted'])
    #print(results)

    #feature importance
    #print(list(zip(train[features], clf.feature_importances_)))
    #print(accuracy_score(test['num'],predictions))


    def grid_search_wrapper(refit_score='precision_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)
                            
        #print(train[features])
        grid_search.fit(train[features], train['num'])

        # make the predictions
        y_pred = grid_search.predict(test[features])

        print('Best params for {}'.format(refit_score))
        print(grid_search.best_params_)

        # confusion matrix on the test data.
        print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
        print(pd.DataFrame(confusion_matrix(test['num'], y_pred), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')

    fin = pd.DataFrame(grid_search_clf.cv_results_)
    fin = fin.sort_values(by='mean_test_precision_score', ascending=False)
    #print(fin.head())

    y_scores = grid_search_clf.predict_proba(test[features])[:, 1]
    p, r, thresholds = precision_recall_curve(test['num'], y_scores)
    #print(thresholds)

    y_pred_adj = adjusted_classes(y_scores, .3)

    print(pd.DataFrame(confusion_matrix(test['num'], y_pred_adj), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    
    def precision_recall_threshold(p, r, thresholds, t=0.5):
        y_pred_adj = adjusted_classes(y_scores, t)

        print(pd.DataFrame(confusion_matrix(test['num'], y_pred_adj), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        # plot the curve
        plt.figure(figsize=(8,8))
        plt.title("Precision and Recall curve ^ = current threshold")
        plt.step(r, p, color='b', alpha=0.2, where='post')
        plt.fill_between(r, p, step='post', alpha=0.2, color='b')
        plt.ylim([0.5, 1.01])
        plt.xlim([0.5, 1.01])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        close_default_clf = np.argmin(np.abs(thresholds - t))
        plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)

    print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head())
    #precision_recall_threshold(p, r, thresholds, 0.3)

def runKNN():
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'n_neighbors' : [1, 5, 10, 15, 20]
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

    clf = KNeighborsClassifier()

    def grid_search_wrapper(refit_score='precision_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        # make the predictions
        y_pred = grid_search.predict(X_test)

        print('Best params for {}'.format(refit_score))
        print(grid_search.best_params_)

        # confusion matrix on the test data.
        print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
        print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')

    fin = pd.DataFrame(grid_search_clf.cv_results_)
    fin = fin.sort_values(by='mean_test_precision_score', ascending=False)
    #print(fin.head())

    y_scores = grid_search_clf.predict_proba(X_test)[:, 1]
    p, r, thresholds = precision_recall_curve(y_test, y_scores)
    #print(thresholds)

    print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_n_neighbors']].round(3).head())

    #knn.fit(X_train, y_train)

    #pred = knn.predict(X_test)

    #print(accuracy_score(y_test, pred))

def runSVM(kernel):
    dataFrame = pd.read_csv('processedclevelandPrime.csv')

    param_grid = {
        'C' : [1, 5, 10]
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

    clf = svm.SVC(kernel = kernel)
    

    def grid_search_wrapper(refit_score='precision_score'):

        skf = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        # make the predictions
        y_pred = grid_search.predict(X_test)

        print('Best params for {}'.format(refit_score))
        print(grid_search.best_params_)

        # confusion matrix on the test data.
        print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
        print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        return grid_search

    grid_search_clf = grid_search_wrapper(refit_score='precision_score')

    fin = pd.DataFrame(grid_search_clf.cv_results_)
    fin = fin.sort_values(by='mean_test_precision_score', ascending=False)
    #print(fin.head())

    y_scores = grid_search_clf.decision_function(X_test)
    p, r, thresholds = precision_recall_curve(y_test, y_scores)
    #print(thresholds)

    print(fin[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_C']].round(3).head())


    #print(clf)
    #clf.fit(X_train, y_train)
    #print(clf.coef_)
    #print(clf.intercept_)
    #predictions = clf.predict(X_test)
    #print(accuracy_score(y_test,predictions))


#runRandomForest()
runKNN()
#runSVM("linear")