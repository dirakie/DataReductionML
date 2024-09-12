import os
import time 
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics 
from sklearn.model_selection import cross_validate
import plotly.express as px
from collections import defaultdict

def cross_validate_classifiers(clf_list, x, y, cv, path, win_metric="accuracy", average="macro"):

    '''Takes a list of scikit classifiers, performs cv for each and stores the results to the given path.
    A dictionary with the best classifier from each split is returned but the storing process takes place for 
    all classifiers.'''
    
    options = ["accuracy", "precision", "recall", "f1"]
    assert win_metric in options, f"Invalid metric. Choose one out of: {options}."

    metric = "test_" + win_metric + "_" + str(average) if win_metric in options[1:] else "test_accuracy"
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    all_clf, best_clf = {}, {}

    for clf in tqdm(clf_list, desc="Benchmarking Progress"):
        print(f"\n\033[1mBuilding {type(clf).__name__}")
        cv_start = time.time()
        scores = cross_validate(clf, x, y, cv = cv, scoring=scoring, return_train_score=True, return_estimator=True, verbose=0, n_jobs=-1)
        print(f"CV elapsed time: {np.round(time.time() - cv_start, 2)} s")
        all_clf[type(clf).__name__] = scores["estimator"]
        best_clf[type(clf).__name__] = all_clf[type(clf).__name__][np.argmax(scores[metric])]

        print(f"Storing CV Results to: {str(path)}")
        estIdx = np.zeros(cv.n_splits)
        estIdx[np.argmax(scores[metric])] = 1
        cv_results = pd.DataFrame(scores).join(pd.DataFrame(estIdx.astype(np.int8),
                                                         columns=["best_estimator"]))
        
        cv_results.to_csv(os.path.join(path, f"{type(clf).__name__}.csv"), index=False)
        #display(cv_results)
        
    return all_clf, best_clf

def evaluate_classifiers(clf_list, x, y, path, win_metric="accuracy", average="macro"):
    
    ''' Takes a list of pre-trained scikit classifiers to make predictions on the test data. The results dataframe includes common
    scikit metrics for evaluation (accuracy, precision, recall, f1-score, testing time, confusion matrix and a classification report string).
    Additionally the best estimator is marked, based on the given win_metric parameter. '''

    options = ["accuracy", "precision", "recall", "f1"]
    assert win_metric in options, f"Invalid metric. Choose one out of: {options}."
    
    metric = "test_" + win_metric + "_" + str(average) if win_metric in options[1:] else "test_accuracy"

    for name, list_of_clf in clf_list.items():
        test_results = defaultdict(list)
        print(f"\n\033[1mTesting best {name}\033[0;0m")
        for clf in list_of_clf:
            t0 = time.time()
            y_pred = clf.predict(x)
            test_time = time.time() - t0
    
            # estimator
            test_results["estimator"].append(clf)
    
            # metrics
            test_results["test_" + "accuracy"].append(metrics.accuracy_score(y, y_pred))
            test_results["test_" + "precision"+ "_" + str(average)].append(metrics.precision_score(y, y_pred, average=average))
            test_results["test_" + "recall"+ "_" + str(average)].append(metrics.recall_score(y, y_pred, average=average))
            test_results["test_" + "f1"+ "_" + str(average)].append(metrics.f1_score(y, y_pred, average=average))
    
            # additional
            test_results["test_" + "time_[s]"].append(test_time)
            test_results["confusion_matrix"].append(metrics.confusion_matrix(y, y_pred))
            test_results["classification_report"].append(metrics.classification_report(y, y_pred))
        
        print(f"Storing Test Results to: {str(path)}\n")
        estIdx = np.zeros(len(clf_list[name]))
        estIdx[np.argmax(test_results[metric])] = 1
        test_results = pd.DataFrame(test_results).join(pd.DataFrame(estIdx.astype(np.int8),
                                                         columns=["best_estimator"]))
        
        test_results.to_csv(os.path.join(path, f"{name}.csv"), index=False)

        # print confusion matrix of best estimator
        cm = test_results[test_results["best_estimator"] == 1]["confusion_matrix"].reset_index(drop=True)[0]
        labels = np.sort(y.unique()).astype(str)
        cmdf = pd.DataFrame(cm, index=labels, columns=labels)
        fig = px.imshow(cmdf, text_auto=True, width = 480, height = 480, color_continuous_scale='Plotly3')
        fig.show(),

        # print classification report of best estimator
        clr = test_results[test_results["best_estimator"] == 1]["classification_report"].reset_index(drop=True)[0]
        print(clr)