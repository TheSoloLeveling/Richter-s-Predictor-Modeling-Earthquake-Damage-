import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pre set values for max cols and chart size
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams["figure.figsize"] = (15,5)

#  Read training data
data=pd.read_csv('Data/train_values.csv')
data.head()


# Read table with target variable
label=pd.read_csv('Data/train_labels.csv')

# Add Target variable to training data
data['damage']=label['damage_grade']

data.info()