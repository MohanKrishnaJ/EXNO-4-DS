# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="700" height="314" alt="image" src="https://github.com/user-attachments/assets/6839c903-8363-4769-914d-091a97df3f57" />

```
df.dropna()
```
<img width="774" height="609" alt="image" src="https://github.com/user-attachments/assets/f793d0a5-02fa-4105-9af9-259395524084" />

```
max_vals = df[['Height', 'Weight']].abs().max()
print(max_vals)
```
<img width="491" height="114" alt="image" src="https://github.com/user-attachments/assets/776d5b8b-516b-4660-ba4c-4f72662f98d9" />

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="775" height="546" alt="image" src="https://github.com/user-attachments/assets/4eeeb4a8-1296-48e1-8c5d-6420d5ebef3d" />

```
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
df[['Height','Weight']]=scalar.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="781" height="538" alt="image" src="https://github.com/user-attachments/assets/6f83d9b5-4186-445f-8d49-fe621c8d9c72" />

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="738" height="622" alt="image" src="https://github.com/user-attachments/assets/3fc3ef76-3d85-4201-bc30-b2987e6c6f63" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="766" height="627" alt="image" src="https://github.com/user-attachments/assets/a9c1347e-6864-4a8d-b0da-ae6d6119dd51" />

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
<img width="739" height="329" alt="image" src="https://github.com/user-attachments/assets/9bab1f96-bada-4783-b5b8-6f59166dae55" />

```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="877" height="314" alt="image" src="https://github.com/user-attachments/assets/134981ce-3964-4214-acd4-e222968e5048" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="413" height="134" alt="image" src="https://github.com/user-attachments/assets/2d3729ee-c655-468a-bef4-86cf262e3dfc" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
```
<img width="626" height="79" alt="image" src="https://github.com/user-attachments/assets/e98fc665-12b5-4ef3-a75c-6f8d68cad67c" />

```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="591" height="102" alt="image" src="https://github.com/user-attachments/assets/39a6501d-fa5b-47b7-bedd-72fe50f0ab7c" />



# RESULT:
Feature scaling and feature selection process has been successfullyperformed on the data set.
    
