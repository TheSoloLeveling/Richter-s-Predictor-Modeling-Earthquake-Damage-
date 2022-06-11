
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train_values.csv')
test = pd.read_csv('test_values.csv')

train_targets = pd.read_csv('train_labels.csv')

# We are keeiping the traningsize of 80% and validation size of 20% of the data.
X_train, X_valid, y_train, y_valid = train_test_split(train, train_targets, train_size=0.8, test_size=0.2, random_state = 42)
num_cols = train._get_numeric_data().columns
object_cols = list( set(train.columns) - set(num_cols))

ordinal_encoder = OrdinalEncoder()
# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

label_X_train.drop('building_id', axis = 1, inplace = True)
label_X_valid.drop('building_id', axis = 1, inplace = True)

from sklearn.ensemble import RandomForestClassifier
#final_model = RandomForestClassifier()

good_params = { 'random_state': 42}
good_params['n_estomators'] = 100
good_params['max_depth'] =  40
good_params['min_samples_split'] =  7
good_params['criterion'] =  'gini'
good_params['max_features'] =  'sqrt'
good_params['max_samples'] =  None
good_params['min_impurity_decrease'] = 0.0
good_params['bootstrap'] = True
good_params['min_samples_leaf'] = 1

#or key, value in good_params.items():
#            setattr(final_model, key, value)

from lightgbm import LGBMClassifier
model_lgbm = LGBMClassifier()
modelGBMC = LGBMClassifier(learning_rate=0.1, max_bin=90, max_depth=30, num_leaves=230, random_state=33)
modelGBMC.fit(label_X_train, y_train.damage_grade)
print(test.info)
train.drop('building_id', axis = 1, inplace = True)
train[object_cols] = ordinal_encoder.transform(train[object_cols])
test.drop('building_id', axis = 1, inplace = True)

predictions = modelGBMC.predict(test)

result=pd.DataFrame(predictions)
result['building_id']=test['building_id']
result.rename(columns={0:'damage_grade'},inplace=True)
result=result[['building_id','damage_grade']]

result.to_csv('resultForest.csv',index=False)
