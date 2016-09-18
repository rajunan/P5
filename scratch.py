# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 21:51:14 2016

@author: Raju
"""

import sys
import pickle
import csv
import numpy as np
sys.path.append("../tools/")
from poi_email_addresses import poiEmails

from feature_format import featureFormat, targetFeatureSplit

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
persons = data_dict.keys()    
#print data_dict.values()
poi=poiEmails()

print len(persons)
print len(poi)

print persons[0]
feature_list = data_dict[persons[0]].keys()

#Dump to csv file
#with open("final_project_dump.csv", "w") as csvfile:
#    writer = csv.DictWriter(csvfile, fieldnames=feature_list)
#    writer.writeheader()
#    writer.writerows(data_dict.values())

#Some of the emails are blank NaN?
_nan = 0
for k,v in data_dict.iteritems():
    a = np.asarray(v.values())
    _nan += len(a[a=="NaN"])
    if v['email_address']=="NaN":
        print "No email for ", k, "poi? ", v['salary']
print float(_nan / (21.0*146))
features_list = ["poi", "salary"]
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
