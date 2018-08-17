#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
import matplotlib.pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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
print len(data_dict)
print len(fieldlist), fieldlist

#how many POIs/non-POIs?
df = pd.DataFrame(data_dict).T
print "POI", df.poi.sum(), "non-POI", df.poi.count()-df.poi.sum()

### explore missing values
#how many fokls have a quantified salary/known email address
df.replace("NaN",np.nan, inplace = True)
for field in fieldlist:
    print "Not missing", field, df.eval(field).count()

### Task 2: Remove outliers

# 'LOCKHART EUGENE E' is a missing datapoint,
# also 'THE TRAVEL AGENCY IN THE PARK' is not somebody's name.

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

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, fieldlist, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
'''
def get_model_results(clf, features_train, labels_train, features_test, labels_test):
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "precision", metrics.precision_score(labels_test, prediction)
    print "recall", metrics.recall_score(labels_test, prediction)
    print "f1_score", metrics.f1_score(labels_test, prediction)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf = GaussianNB()
get_model_results(clf, features_train, labels_train, features_test, labels_test)
'''

'''
clf = DecisionTreeClassifier()
get_model_results(clf, features_train, labels_train, features_test, labels_test)
'''

'''
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

parameters = {'anova__k':(2,3,4,5,6,7,8,9,10),
                'dtree__criterion':('gini', 'entropy'),
                'dtree__min_samples_leaf':(1,2,3,4,5),
                'dtree__min_samples_split':(2,3,4,5),
                'dtree__max_depth':(2,3,4,5,None)}
dtree = DecisionTreeClassifier()

cv = StratifiedShuffleSplit(labels, 1000, test_size=0.3, random_state=42)
anova_filter = SelectKBest()
dtree = DecisionTreeClassifier(f_classif)

Pipe = Pipeline([('anova', anova_filter),('dtree',dtree)])
#Pipe.set_params(anova_k=10,dtree_min_samples_split=2)

#score_list = ['precision', 'recall', 'f1']
grid = GridSearchCV(Pipe, parameters, cv=cv, scoring='recall', n_jobs=2)
grid.fit(features, labels)
print grid.best_estimator_
print grid.best_score_

features_list = ['poi', ]
for i in range(len(fieldlist)-1):
    if grid.best_estimator_.named_steps['anova'].get_support()[i] == True:
        features_list.append(fieldlist[i+1])

#the final classifier after autotuned:
clf = grid.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
