
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import RobustScaler


from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Read training data
data=pd.read_csv('train_values.csv')
X_test = pd.read_csv('test_values.csv')
label=pd.read_csv('train_labels.csv')

# Add Target variable to training data
data['damage']=label['damage_grade']


# Define X and y variables
X_train = pd.get_dummies(data.loc[:,:'has_secondary_use_other'])

Y_train =data['damage'].astype(int)

"""""
log_clf= LogisticRegression()
filename = 'finalized_model.sav'
pickle.dump(log_clf, open(filename, 'wb'))


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg_cv = GridSearchCV(log_clf,grid,cv=10)
logreg_cv.fit(X_train,Y_train)

print (logreg_cv.best_params_)
print (logreg_cv.best_score_)
"""""

logreg2 = LogisticRegression(C=0.001,penalty="l2")

predictions = logreg2.fit(X_train,Y_train).predict(pd.get_dummies(X_test))

result=pd.DataFrame(predictions)
result['building_id']=X_test['building_id']
result.rename(columns={0:'damage_grade'},inplace=True)
result=result[['building_id','damage_grade']]

result.to_csv('result.csv',index=False)
