
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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
# Read training data
data=pd.read_csv('train_values.csv')
X_test = pd.read_csv('test_values.csv')
label=pd.read_csv('train_labels.csv')

# Add Target variable to training data
data['damage']=label['damage_grade']


# Define X and y variables
X_train = pd.get_dummies(data.loc[:,:'has_secondary_use_other'])

Y_train =data['damage'].astype(int)
X_trainn, X_valid, y_train, y_valid = train_test_split(X_train, label, train_size=0.8, test_size=0.2, random_state = 42)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

model = Sequential()
model.add(Dense(107, input_shape = (X_train.shape[1],), activation = 'relu'))
model.add(Dense(53, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [f1_m])

predictions = model.fit(X_trainn, Y_train, batch_size = 96, epochs = 30, validation_data = (X_valid, y_valid)).predict(pd.get_dummies(X_test))
#logreg2 = LogisticRegression(C=0.001,penalty="l2")

#predictions = logreg2.fit(X_train,Y_train).predict(pd.get_dummies(X_test))


result=pd.DataFrame(predictions)
result['building_id']=X_test['building_id']
result.rename(columns={0:'damage_grade'},inplace=True)
result=result[['building_id','damage_grade']]

result.to_csv('result.csv',index=False)
