#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
import numpy as np
### Task 1: Select what features you'll use.

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

fieldlist = list(data_dict["SKILLING JEFFREY K"].keys())
fieldlist.remove('email_address')
fieldlist.remove('poi')
fieldlist.insert(0, 'poi')

data = featureFormat(data_dict, fieldlist, sort_keys = True)

### describe the dataset brifely
# the number of data points
print len(data_dict), list(data_dict.keys())
print len(fieldlist), fieldlist

#how many POIs/non-POIs?
poi_n, nonpoi_n = 0, 0
for k in data_dict.iterkeys():
    if data_dict[k]["poi"]==1:
        poi_n = poi_n + 1
    elif data_dict[k]["poi"]==0:
        nonpoi_n = nonpoi_n + 1
print "POIs", poi_n, "non-POIs", nonpoi_n

### explore missing values
#how many fokls have a quantified salary/known email address
salary, email_address = 'salary', 'email_address'
def getfilled(field):
    count=0
    for name in data_dict.keys():
        if data_dict[name][field] != 'NaN':
            count +=1
    return count
def getunfilled(field):
    count=0
    for name in data_dict.keys():
        if data_dict[name][field] == 'NaN':
            count +=1
    return count
print "number of quantified salaries", getfilled(salary)
print "number of known email address", getfilled(email_address)

for field in fieldlist:
    print "number of filled data points for", field, getfilled(field)

### Task 2: Remove outliers
### read in data dictionary, convert to numpy array

def getmissing_datapoint(fieldlist):
    missing_datapoint = []
    for name in data_dict.keys():
        count = 0
        for field in fieldlist:
            if data_dict[name][field] == 'NaN':
                count +=1
        if count >= 20:
            missing_datapoint.append(name)
    print missing_datapoint

getmissing_datapoint(fieldlist)

# 'LOCKHART EUGENE E' is a missing datapoint,
# also 'THE TRAVEL AGENCY IN THE PARK' is not somebody's name.

import matplotlib.pyplot

data_dict.pop( "TOTAL", 0 )
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

'''
features = ["salary", "bonus", ]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
'''

### Task 3: Create new feature(s)
def getPropotion(poi_messages, all_messages):
    propotion = 0.
    if poi_messages == 'NaN' or all_messages == 'NaN' or all_messages == 0:
        propotion = 'NaN'
    else:
        propotion = float(poi_messages)/float(all_messages)
    return propotion

for ele in data_dict.values():
    ele['propotion_from_poi'] = getPropotion(ele['from_poi_to_this_person'], \
    ele['to_messages'])
    ele['propotion_to_poi'] = getPropotion(ele['from_this_person_to_poi'], \
    ele['from_messages'])

fieldlist.append('propotion_from_poi')
fieldlist.append('propotion_to_poi')

my_dataset = featureFormat(data_dict, fieldlist, sort_keys = True)
labels, features = targetFeatureSplit(my_dataset)

### Use SelectKBest to automatically select features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit

print len(features[1])
select_f = SelectKBest(f_classif, k=10).fit(features, labels)
autoselected_features = {}
sefeatures_score = []
for index, value in enumerate(select_f.get_support().flat):
    if value:
        autoselected_features[fieldlist[index + 1]] = select_f.scores_[index]
sortby_score = sorted(autoselected_features.items(),key=lambda x:x[1], reverse=True)
for ele in sortby_score:
    print ('{0} {1}'.format
        (ele[0],ele[1]))


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'total_stock_value', 'bonus', 'salary', 'propotion_to_poi', 'exercised_stock_options'] # You will need to use more features

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def get_model_results(clf, features_train, labels_train, features_test, labels_test):
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "precision", metrics.precision_score(labels_test, prediction)
    print "recall", metrics.recall_score(labels_test, prediction)
    print "f1_score", metrics.f1_score(labels_test, prediction)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
get_model_results(clf, features_train, labels_train, features_test, labels_test)
'''
'''
clf = DecisionTreeClassifier()
get_model_results(clf, features_train, labels_train, features_test, labels_test)
'''
'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
get_model_results(clf, features_train, labels_train, features_test, labels_test)
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
'''
parameters = {'criterion':('gini', 'entropy'), 'max_features':(2,3,4,5,None),\
                'min_samples_split':(2,3,4,5), 'max_depth':(2,3,4,5,6,7,8,9,10,None)}
dtree = DecisionTreeClassifier()

clf = GridSearchCV(dtree, parameters)
get_model_results(clf, features_train, labels_train, features_test, labels_test)
print clf.best_estimator_
'''

#the final classifier after autotuned:
clf = DecisionTreeClassifier(criterion='gini', max_features=2, min_samples_split=2, max_depth=2)

'''
#cross validation
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
feature_importances = np.zeros(len(features_list)-1)
precision = []
recall = []
f1 = []
for train_index, test_index in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for i in train_index:
        features_train.append(features[i])
        labels_train.append(labels[i])
    for j in test_index:
        features_test.append(features[j])
        labels_test.append(labels[j])
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    precision.append (metrics.precision_score(labels_test, pred))
    recall.append( metrics.recall_score(labels_test, pred))
    f1.append( metrics.f1_score(labels_test, pred))
    feature_importances = feature_importances + np.array(clf.feature_importances_)
mean_feature_importances = feature_importances / 1000
print 'Mean feature importances:'
print sorted(zip(map(lambda x: round(x, 4), mean_feature_importances), features_list[1:]), reverse=True)
print 'precision: %f, recall: %f, f1: %f' %(np.mean(precision), np.mean(recall), np.mean(f1))
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
