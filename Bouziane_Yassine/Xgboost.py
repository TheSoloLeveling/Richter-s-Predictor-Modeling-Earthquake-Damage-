import pandas as pd
import numpy as np
import seaborn as sns


from xgboost import XGBClassifier
import xgboost as xgb

import cudf
import pynvml

pdata=pd.read_csv('train_values.csv')
data=cudf.from_pandas(pdata)
data.head()