# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:05:22 2019

@author: ajadhav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


df = pd.read_csv("./data/SalaryData.csv")

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

df_copy = train_set.copy()

train_set_full = train_set.copy()

train_set = train_set.drop(["Salary"], axis=1)

train_labels = df_copy["Salary"]

lin_reg = LinearRegression()

lin_reg.fit(train_set, train_labels)

joblib.dump(lin_reg, "linear_regression_model.pkl")

#import requests
#BASE_URL = "http://localhost:5000"
#
#years_exp = {"yearsOfExperience": 8}
#
#response = requests.post("{}/predict".format(BASE_URL), json = years_exp)
#
#response.json()
