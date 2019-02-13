# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 2019

@author: muehl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

used_columns= ['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
'HeatingQC','GrLivArea','FullBath','KitchenQual','TotalBsmtSF',
'MiscVal','SaleCondition','GarageCars','GarageArea']
cleanup_nums = {"HeatingQC":     {"Ex": 5, "TA": 3,"Gd": 4, "Fa": 2,"Po": 1},
                "SaleCondition": {"Normal": 5, "Partial": 3,"Abnorml": 4, "Family": 2,"Alloca": 1,"AdjLand": 0},
                "KitchenQual": {"Ex": 5, "TA": 3,"Gd": 4, "Fa": 2,"Po": 1}}
# Import Data

df_train = pd.read_csv('Data/train.csv')
df_train.replace(cleanup_nums, inplace=True)
same_lotsize = df_train[(df_train['LotArea'] > 3000) &( df_train['LotArea'] < 5000) ]
#sns.distplot(df_train['SalePrice']);

#scatter plot grlivarea/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'KitchenQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);