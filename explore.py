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

"""""
# Check correlation of all the columns to see if something stands-out explicitly
corr=data.corr()
plt.subplots(figsize=(10,10))

sns.heatmap(corr, xticklabels=True,yticklabels=True,  vmin=0, vmax=1,
    cmap=sns.diverging_palette(220, 20, n=100),
    square=True)


data['damage'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Damage Grade')
plt.ylabel('No. of records')
plt.title('Target distribution')


data['count_floors_pre_eq'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Damage Grade')
plt.ylabel('No. of records')
plt.title('Distribution of count_floors_pre_eq variable')


data['age'].value_counts().sort_index().plot.bar()

sns.countplot(x='height_percentage',data=data, hue='damage');


a=data.groupby(['height_percentage','damage']).size().reset_index()
a.head()
b=a.pivot(index='height_percentage',columns='damage',values=0).reset_index()
print(b.head())
b.set_index('height_percentage',inplace=True)
b.plot(kind='bar',stacked=True,title='Type of damage by height percentage of buildings');
"""""


data['count_families'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Damage Grade')
plt.ylabel('No. of records')
plt.title('Distribution of count_families variable')
plt.show()

