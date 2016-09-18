#!/usr/bin/python

# Key to highlight is that choice of cross validation 
# drives outcome of feature selections and peformance
# As such I worked with both train_test_split and StratifiedShuffleSplit leading up
# selecting feature list and also tuning my algorithm

_verbose = 0  # Can do a better job at logging but did not bother for this excercise
# =0 shortlisted feature set and tester output is logged
# =1 +outputs detailed feature selection iterations
# =2 +outputs the various classifiers (with default params) runs incl SVM tuning
# =3 +outputs the scatterplots used to review features
# =4 +outputs tuning, WARNING: can take up to 300 seconds

import numpy
import copy
import sys
import pickle
import pprint
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from my_helper_routines import plot_scatter, outlier_key, dict_column  

#Use cv = train test split
def review_features_with_tts(my_dataset, features_list, min_samples_split=10):
    print "... reviewing features using cv train test split", features_list[1:]
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    dt.fit(features_train, labels_train)
    #Note: features exclude POI and hence we start from features[1:]
    #print dt.feature_importances_
    fi = zip(features_list[1:], [round(elem,3) for elem in dt.feature_importances_])
    print "... feature imp descending: ", sorted(fi, key = lambda tup: tup[1], reverse = True)
    labels_predict = dt.predict(features_test)
    print "... Accuracy: ",  round(accuracy_score(labels_test, labels_predict),3), \
          "Precision: ", round(precision_score(labels_test, labels_predict),3), \
          "Recall: ",    round(recall_score(labels_test, labels_predict),3), \
          "F1: ",        round(f1_score(labels_test, labels_predict),3)

#Use cv = StratifiedShuffleSplit
def review_features_with_sss(my_dataset, features_list, min_samples_split=10):
    print "... reviewing features using cv stratified shuffle split", features_list[1:]
    clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
    test_classifier(clf, my_dataset, features_list) #too lazy to write my own SSS
    #Note: features exclude POI and hence we start from features[1:]
    #print clf.feature_importances_
    fi = zip(features_list[1:], [round(elem,3) for elem in clf.feature_importances_])
    print "... feature imp descending: ", sorted(fi, key = lambda tup: tup[1], reverse = True)

def do_selectKBest(features_list, my_dataset, k=3):
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    #Add dummy rows to identify feature set suggested by selectKBest
    labels.insert(0,0); features.insert(0,range(len(features_list)-1))
    features_new = SelectKBest(chi2,k).fit_transform(features, labels)
    selected_features = []
    for i in features_new[0]:
        selected_features.append(features_list[1:][int(i)])
    return ["poi"]+selected_features #Need to add back "poi"

# Helper routine to call udacity tester procedure and dumping elapsed time
def try_clf(clf, my_dataset, features_list):
    t0 = time()
    test_classifier(clf, my_dataset, features_list)
    print "... training time:", round(time()-t0,3),"s"

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',                    #Base data
                 'bonus',                     #Base data  
                 'other',                     #Seems to correlate for POIs             
                 'long_term_incentive',       #This should be higher for executives to avoid incentivizing gaming for short term gains
                 'exercised_stock_options',   #given POIs were those with insider info, this seems like a crucial feature
                 'shared_receipt_with_poi'    #if poi shares lot of emails to this person then this person could be also a poi
                 ]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Data exploration, Outlier investigation
#What is the size of the dataset?
persons = list(data_dict.keys())
features = data_dict[persons[0]].keys()
rec_len = len(persons) #Number of records
feature_size = len(features) #Feature set size
print "Number of records: ", rec_len
print "Number of features: ", feature_size

#Some of the data points are NaN, how many?
_nan = 0
poi = []
for k,v in data_dict.items():
    a = numpy.asarray(v.values())
    _nan += len(a[a=="NaN"])
    poi.append(float(v["poi"]))
print "Ratio of NaN in the data set: ", \
    round(_nan / (float(rec_len*feature_size)),2)
print "Number of POIs in the data set: ", sum(poi), " which is ", round(sum(poi)/len(poi),2)
#Ratio of NaN is 44% which is very high and only 12% of the persons are POI

### Task 2: Remove outliers ###################################################
# Dump to CSV file for manual review
#import csv
# with open("final_project_dump.csv", "w") as csvfile:
#    writer = csv.DictWriter(csvfile, fieldnames=feature_list)
#    writer.writeheader()
#    writer.writerows(data_dict.values())
# Imported into Excel (see writeup for "offline" analysis)
# Reviewed outliers in features with most data points available: total payments & total stock value) 
# and based on that removed TOTAL and THE TRAVEL AGENCY IN THE PARK accordingly
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
# Should I pop Lockhart Eugene as well given he has no data available at all?
persons = data_dict.keys()
print "Number of records post removal of outliers: ", len(persons)

#Visualize relationship b/w some of the features
if _verbose > 2:
    print "\n", "#############################################################"
    print "Features review ..."
    # 2.1 Bonus vs salary
    # ? Are POIs on higher bracket in terms of compensation
    f1="salary"; f2="bonus"
    plot_scatter(f1, f2, data_dict, check_ratio=True)
    ### => Reject: not enough discrimation power, POIs are not unfairly overcompensated
    
    # 2.2 Compensation vs exercised_stock_options
    # ? Are POIs those who have higher excercised stock options compared to salary/bonus
    f1="bonus"; f2="exercised_stock_options"
    plot_scatter(f1, f2, data_dict)
    plot_scatter(f1, f2, data_dict, check_ratio=True, y_scale=[0,4])
    f1="salary"; f2="exercised_stock_options"
    plot_scatter(f1, f2, data_dict)
    # Salary outlier if ratio is computed =>  'BANNANTINE JAMES M'
    plot_scatter(f1, f2, data_dict, check_ratio=True, y_scale=[0,10])
    # ? is composite feature of ratio to bonus+salary better
    for k,v in data_dict.items():
        data_dict[k]["totComp"] = \
            float(data_dict[k]["salary"])+ \
            float(data_dict[k]["bonus"])+\
            float(data_dict[k]["other"])
    f1="totComp"; f2="exercised_stock_options"
    plot_scatter(f1, f2, data_dict, check_ratio=True)
    plot_scatter(f1, f2, data_dict, check_ratio=True, y_scale=[0,3])

    ### => Given poor data quality, composite feature of bonus+salary seems useful

    #2.3 Incentives vs exercised stock options
    # ? Intuitively long term incentive should be less attacive to trade i.e. sale of option unless POI  
    f1="long_term_incentive"; f2="exercised_stock_options"
    plot_scatter(f1, f2, data_dict, check_ratio=True)
    plot_scatter(f1, f2, data_dict, check_ratio=True, y_scale=[0.5,12])
    ### => Interestingly those with less LTI seem to have higher ratio than POI
    ### On to some thing, will use this as new feature
    
    #2.4 Restricted stock vs exercised stock options
    # ? Higher amount of restricted stock should allocated to POIs
    f1="restricted_stock"; f2="exercised_stock_options"
    plot_scatter(f1, f2, data_dict, check_ratio=True,y_scale=[0,10])
    ### => 1: Not all POI are have higher restricted stock
    ### => 2: Ratio also is not conclusive; I would have expected higher ratios
    
    # 2.5 ? Emails from and to POIs or shared_receipt_with_poi
    f1='from_poi_to_this_person'; f2='from_this_person_to_poi'
    plot_scatter(f1, f2, data_dict)
    plot_scatter(f1, f2, data_dict,x_scale=[0,100],y_scale=[0,100])
    ### => POIs seem to be talking to POIs more often hence will keep this feature for further review
    ### => However many non POIs talk to POIs hence will discard this feature
    f1='shared_receipt_with_poi'; f2='from_poi_to_this_person'
    plot_scatter(f1, f2, data_dict)
    plot_scatter(f1, f2, data_dict, check_ratio=True)
    ### => Pattern emerging with ratio of poi to a person by shared receipts with poi
    
### Task 3: Create new feature(s) #################################################3
# Explore possible new features
if _verbose > 0:
    print "\n", "#############################################################"
    print "Feature selection and review composite ones..."

import math

# 3.1. Total comp (sal, bonus, other)
nf = "totComp"
for k in data_dict:
    totComp = float(data_dict[k]["salary"]) + \
              float(data_dict[k]["bonus"]) + \
              float(data_dict[k]["other"])
    if math.isnan(totComp): totComp = 0.0
    data_dict[k][nf] = totComp       
        
# 3.2. Exercised stock options related to total comp
nf = "exerOpt_to_totComp"
for k in data_dict:
    r = float(data_dict[k]["exercised_stock_options"])/ \
           sum([float(data_dict[k]["salary"]),
                float(data_dict[k]["bonus"]),
                float(data_dict[k]["other"])])
    if math.isnan(r): r = 0.0
    data_dict[k][nf] = r

# 3.3. Exercised stock options related to long term incentive
nf = "exerOpt_to_longIncen"
for k in data_dict:
    r = float(data_dict[k]["exercised_stock_options"])/ \
        float(data_dict[k]["long_term_incentive"])
    if math.isnan(r): r = 0.0
    data_dict[k][nf] = r

# 3.4. from POI to this person by shared receipt with POI 
nf = "from_POI_by_shared_receipt"
for k in data_dict:
    r = float(data_dict[k]["from_poi_to_this_person"])/ \
        float(data_dict[k]["shared_receipt_with_poi"])
    if math.isnan(r): r = 0.0
    data_dict[k][nf] = r

# My set of new features
features_composite_list = ["totComp", "exerOpt_to_totComp", 
                           "exerOpt_to_longIncen", "from_POI_by_shared_receipt"]

#Select features list 1 using cv = train test split
features_list_tts = copy.deepcopy(features_list) 
if _verbose > 0: 
    print "\n\n","TTS Iteration 1 (base features reviewed): "
    review_features_with_tts(data_dict, features_list_tts)
features_list_tts += features_composite_list
if _verbose > 0: 
    print "\n","TTS Iteration 2 (add composite features): "
    review_features_with_tts(data_dict, features_list_tts)
features_list_tts.remove("bonus");
features_list_tts.remove("salary");
features_list_tts.remove("long_term_incentive")
features_list_tts.remove("exerOpt_to_totComp")
features_list_tts.remove("from_POI_by_shared_receipt")
features_list_tts.remove("other");
if _verbose > 0: 
    print "\n","TTS Iteration 3 (removed all features with near zero importances): "
    review_features_with_tts(data_dict, features_list_tts)
    
#Select features list 2 using cv = stratified_shuffle_split
features_list_sss = copy.deepcopy(features_list) 
if _verbose > 0: 
    print "\n\n","SSS Iteration 1 (base features reviewed): "
    review_features_with_sss(data_dict, features_list_sss)
features_list_sss += features_composite_list
if _verbose > 0: 
    print "\n","SSS Iteration 2 (add composite features): "
    review_features_with_sss(data_dict, features_list_sss)
features_list_sss.remove("bonus")
features_list_sss.remove("salary")
features_list_sss.remove("exerOpt_to_longIncen")
features_list_sss.remove("from_POI_by_shared_receipt")
features_list_sss.remove("long_term_incentive")
features_list_sss.remove("shared_receipt_with_poi")
if _verbose > 0: 
    print "\n","SSS Iteration 3 (removed all features with near zero importances): "
    review_features_with_sss(data_dict, features_list_sss, min_samples_split=10)

# Select features list 3 Validate my views using SelectkBest 
# Note: I tried several k(=2,3,4) and found 2 to be the best in performance
features_list_kBest = []
features_list_kBest = features_list + features_composite_list
features_list_kBest_k_4 = do_selectKBest(features_list_kBest, data_dict, k=4)
features_list_kBest_k_3 = do_selectKBest(features_list_kBest, data_dict, k=3)
features_list_kBest_k_2 = do_selectKBest(features_list_kBest, data_dict, k=2)

#Basic performance of each features list reviewed
if _verbose > 0:
    print "\n", "Basic peformance for each of features list and their relative importance ..."
    print "\n","... Features list 1 performance of my list using tts ...", features_list_tts[1:]
    review_features_with_sss(data_dict, features_list_tts)
    # Accuracy: 0.85036 Precision: 0.47380 Recall: 0.42950 F1: 0.45056
    print "\n", "... Features list  2 performance of my list using sss ...", features_list_sss[1:] 
    review_features_with_sss(data_dict, features_list_sss)
    # Accuracy: 0.84957 Precision: 0.47120 Recall: 0.43350 F1: 0.45156
    print "\n", "... Features list 3 performance of kBest = 4 ", features_list_kBest_k_4
    review_features_with_sss(data_dict, features_list_kBest_k_4)
    # 
    print "\n", "... Features list  4 performance of kBest = 3", features_list_kBest_k_3
    review_features_with_sss(data_dict, features_list_kBest_k_3)
    # 
    print "\n", "... Features list 5 performance of kBest = 2", features_list_kBest_k_2
    review_features_with_sss(data_dict, features_list_kBest_k_2)

# Seems like selectKBest is best choice but high accuracy and precision
# However, worrying that I am overfitting
print "\n", "#############################################################"
print "\n", "Shortlisted features list ..."
features_list = features_list_kBest_k_2
pprint.pprint(features_list)
print "\n", "#############################################################"

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

if _verbose > 1:
    print "\n", "#############################################################"
    print "Trying varity of classifiers ..."

    print "\n","clf = ExtraTreesClassifier ..."
    clf = ExtraTreesClassifier()
    try_clf(clf, my_dataset, features_list)

    print "\n","k-Nearest ..."
    clf = KNeighborsClassifier()
    try_clf(clf, my_dataset, features_list)

    print "\n","SVM default and no scaling ..."
    clf = SVC()
    try_clf(clf, my_dataset, features_list)
    print "...SVM tuning ..."
    from sklearn.preprocessing import MinMaxScaler
    #data = featureFormat(my_dataset, features_list_kBest_k_4, sort_keys = True)
    #data = featureFormat(my_dataset, features_list_kBest_k_2, sort_keys = True)
    print "\n","... switching to different feature test for SVM"
    print features_list_kBest_k_3
    data = featureFormat(my_dataset, features_list_kBest_k_3, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features = MinMaxScaler().fit_transform(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    param_grid = {'kernel': ['rbf'],
                  'C': [10, 500,1000, 2000, 5000, 50000],
                  'gamma': [0.000001, 0.001, 0.01, 0.1, 0.5, 1, 10, 11, 20]} 
    svr = SVC()
    clf = GridSearchCV(svr, param_grid)
    clf.fit(features_train, labels_train)
    print clf.best_estimator_
    labels_predict = clf.predict(features_test)
    print "... Accuracy: ",  round(accuracy_score(labels_test, labels_predict),3), \
        "Precision: ", round(precision_score(labels_test, labels_predict),3), \
        "Recall: ",    round(recall_score(labels_test, labels_predict),3), \
        "F1: ",        round(f1_score(labels_test, labels_predict),3)
    # SVC(C=5000, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape=None, degree=3, gamma=20, kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    # ...Accuracy:  0.892 Precision:  0.5 Recall:  1.0 F1:  0.667

if _verbose == 0:
    print "Checking out SVM testing with scale incl via pipeline ..."
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    param_grid = {'kernel': ['rbf'],
                  'C': [10, 500,1000, 2000, 5000, 50000],
                  'gamma': [0.000001, 0.001, 0.01, 0.1, 0.5, 1, 10, 11, 20]} 
    svr = SVC()
    gridCV = GridSearchCV(svr, param_grid)
    clf = make_pipeline(MinMaxScaler(),gridCV)  
    try_clf(clf, my_dataset, features_list_kBest_k_3)
    print gridCV.best_estimator_
'''
    Accuracy: 0.84277
    Precision: 0.46707      Recall: 0.15600
    F1: 0.23388     F2: 0.17997
    Total predictions: 13000        True positives:  312    False positives:  356   False negatives: 1688   True negatives: 10644

    training time: 545.776 s
    SVC(C=500, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma=0.5, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
'''
'''
    suggested code from instuctor
    svm_clf = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('classifier', SVC())
                    ])
    svm_clf.fit(features_train,labels_train)
    accuracy = svm_clf.score(features_test,labels_test)
    predicted = svm_clf.predict(features_test)
    output = test_classifier(svm_clf, my_dataset, features_list)
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

if _verbose > 3: #
    print "\n", "#############################################################"
    print "Tunning ... "
    print "Warning can take up to 300s" 

    print "\n","k Nearest Neighbor tuning ..."
    " ... using sss"    
    svr = KNeighborsClassifier()
    param_grid = {'n_neighbors': [2,3,4,5,10],
                  'leaf_size':[1,2,3,20,30],
                  'weights': ['uniform','distance']}
    clf = GridSearchCV(svr, param_grid)
    print " ... using sss"
    try_clf(clf, my_dataset, features_list)
    print "... best estimator: :", clf.best_estimator_
    #   Results of run
    #   Accuracy: 0.87183 Precision: 0.71670 Recall: 0.38200 F1: 0.52518 F2: 0.49837
    #   KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
    #                 metric_params=None, n_jobs=1, n_neighbors=3, p=2,
    #                 weights='uniform')
    #   time > 300s

    print "\n"," ... using tts"
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    param_grid = {'n_neighbors': [3,4,5,6,7,8,9,10],
                  'leaf_size':[1,2,3,4,5,6,7,8,9,10],
                  'weights': ['uniform','distance']}
    clf = GridSearchCV(svr, param_grid)
    clf.fit(features_train, labels_train)
    labels_predict = clf.predict(features_test)
    print "... Accuracy: ",  round(accuracy_score(labels_test, labels_predict),3), \
    "Precision: ", round(precision_score(labels_test, labels_predict),3), \
    "Recall: ",    round(recall_score(labels_test, labels_predict),3), \
    "F1: ",        round(f1_score(labels_test, labels_predict),3)
    print "... best estimator: :", clf.best_estimator_
    #   Results of run
    #   Accuracy:  0.889 Precision:  0.667 Recall:  0.4 F1:  0.5
    #   KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
    #           metric_params=None, n_jobs=1, n_neighbors=4, p=2,
    #           weights='distance')
    #   time 1s


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

_n=3
_l=1
_w='uniform'
clf = KNeighborsClassifier(n_neighbors=_n, leaf_size=_l, weights=_w)
print "\n", "Final run ..."
try_clf(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)