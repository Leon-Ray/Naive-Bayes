# --------------------
# Purpose: Classify crimes in San Francisco using a simple Naive Bayes algorithm.
# Author: Leon Raykin
# Python Version: 2.7
# --------------------

import pandas as pd
import numpy as np

###train data supervised learning###

#read train data
train = pd.read_csv('train.csv')

#calculate prior probabilities
cat = train['Category']
prior_prob = cat.value_counts(normalize=True) 
prior_prob = pd.DataFrame(prior_prob, columns=['PriorProbability'])

#calculate likelihoods
train['Hour'] = pd.to_datetime(train['Dates']).apply(lambda x: x.hour) #create hour of day column
likelihood_hour = pd.crosstab(train.Category, train.Hour).apply(lambda r: r/r.sum(), axis=1) #time (hour of day)
likelihood_dow = pd.crosstab(train.Category, train.DayOfWeek).apply(lambda r: r/r.sum(), axis=1) #time (day of week)
likelihood_district = pd.crosstab(train.Category, train.PdDistrict).apply(lambda r: r/r.sum(), axis=1) #location (neighborhood)

#function to estimate probabilities of each category and return class label
def classify(row):
    
    #attributes used for classification
    hour = row['Hour']
    dow = row['DayOfWeek']
    district = row['PdDistrict']

    #likelihoods for the given attributes
    likelihood_hour_sub = likelihood_hour[hour]
    likelihood_dow_sub = likelihood_dow[dow]
    likelihood_district_sub = likelihood_district[district]

    #probabilities dataframe for the given attributes
    probabilities = prior_prob.join(likelihood_hour_sub, how="inner").join(likelihood_dow_sub, how="inner").join(likelihood_district_sub, how="inner")
    probabilities['Probability'] = probabilities.prod(axis=1) #doesn't include denominator, since it's the same across all classes
    probabilities = probabilities.sort('Probability', ascending=False)

    #class label based on the highest probability
    class_label = probabilities.index.values[0]
    return(class_label)


###test data classification###

#read test data
test = pd.read_csv('test.csv')

#add a new column to the test data with the class label
test['Hour'] = pd.to_datetime(test['Dates']).apply(lambda x: x.hour) #create hour of day column
test['ClassLabel'] = test.apply(lambda row: classify(row), axis=1)
#test.to_csv('test_classified.csv', index=False)

#produce a crosstab of the results
results = pd.crosstab(test['Id'], test['ClassLabel'])
results = pd.DataFrame(results)

#produce a new dataframe that includes columns for all categories (Kaggle submission format) 
cats = prior_prob.index.values
submission = pd.DataFrame(index=test['Id'], columns=cats)
submission = submission.fillna(0)
submission = (submission + results).fillna(0)

#write submission
submission.to_csv('submission.csv')

