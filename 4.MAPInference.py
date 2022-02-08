# -*- coding: utf-8 -*-
"""
         Homework #5: MAP Inference
@author: Adrian Ramos 

"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#%% Dataset exploration
df = pd.read_table('fruit_data_with_colors.txt')
print(df.describe())

#%% Training set and Test set creation
x_t = train_test_split(x,y)

#%% Logistic Regression model

logr_model = LogisticRegression()
logr_model.fit(x_t,y_t)


#%% NAIVE BAYES MODEL

gnb_model = GaussianNB()
gnb.fit(x_t,y_t)