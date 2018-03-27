# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
""" The iloc function for a DataFrame means localization by integer"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#taking care of missing data
"""from sklearn.preprocessing import Imputer
#this object works for substitute all the values that are missing
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
#this method is applyed for all the lines in our dataframe but only for 
#columns that are into the 1:3 interval (the 3 is not inside the 
#interval) making only a mapping
imputer = imputer.fit(X[:,1:3])
#this method apply what the fit mapped 
X[:,1:3] = imputer.transform(X[:,1:3])
"""
""" Applying labels on text columns changing """
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
#this changes all the text data into the column in numerical categories
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
#this will avoid the number of the categories automatically changed to 
#be evalued like one greater than another and in this case they aren't
oneHotEncoder = OneHotEncoder(categorical_features= [0])
#This splits one categorical column in N new features, one for each category
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
"""


""" Splitting the dataset into the Training set and Test set."""
from sklearn.cross_validation import train_test_split
#this defines how much from the traing data should be used as test set, through text_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


"""Feature scalling brings all the features to a range between -1 to +1 so 
there are no more dominants features, which makes all the features to be
in the same level of importance, so the euclidian distance will be 
considerated by all the features. This could be make by standadization or
normalization"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#we SHOULD first fit_transform for the training set so we can define the range
# which we will work for training, test and crossvaleu set
X_train = sc_X.fit_transform(X_train)
#then ONLY apply TRANSFORM to the test set, so all of then will became same range
X_test = sc_X.transform(X_test)
#we don't need to apply the feature scale to y, because it is a categorical column (0,1 values)
