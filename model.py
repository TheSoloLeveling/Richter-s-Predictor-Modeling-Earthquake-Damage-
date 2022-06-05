import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate,KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Pre set values for max cols and chart size
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams["figure.figsize"] = (15,5)

# Read training data
data=pd.read_csv('Data/train_values.csv')
data.head()

# Read table with target variable
label=pd.read_csv('Data/train_labels.csv')
label.head()

# Add Target variable to training data
data['damage']=label['damage_grade']

# Check distribution of target variable to find out if there is class imbaance problem
data['damage'].value_counts()

# Define X and y variables
X=pd.get_dummies(data.loc[:,:'has_secondary_use_other'])
le = LabelEncoder()

y=data['damage'].astype(int)
#y= le.fit_transform(y) for python 3.10

# Parameters for XGboost
n_jobs=[-1]
n_estimators=np.arange(100,1000,100)
learning_rate=[0.03,0.01,0.1]
max_depth=np.arange(10,100,15)

# Param grid for Xgboost
param_grid={'n_jobs':n_jobs,
            'n_estimators':n_estimators,
            'max_depth':max_depth,
            'learning_rate':learning_rate
           }

# Fit scaled traing data on XGboost Classifier- This step took 17 hours to complete
clf=XGBClassifier()
kf=KFold(n_splits=2,shuffle=True)
rs=RandomizedSearchCV(clf,param_distributions=param_grid,cv=kf,scoring='f1_micro')
rs.fit(X,y )




