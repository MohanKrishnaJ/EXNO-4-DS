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
<img width="826" height="393" alt="image" src="https://github.com/user-attachments/assets/7a808040-ebec-4377-ab6e-3a503663f184" />

```
df.dropna()
```
<img width="589" height="618" alt="image" src="https://github.com/user-attachments/assets/912f137c-9c27-4eae-8c51-32b64dee67d5" />

```
max_vals = df[['Height', 'Weight']].abs().max()
print(max_vals)
```
<img width="705" height="104" alt="image" src="https://github.com/user-attachments/assets/7639070d-f51a-4647-a40b-eba27713f7a9" />

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="857" height="543" alt="image" src="https://github.com/user-attachments/assets/43b5c967-c34a-45e8-b86c-f7dba6f381de" />

```
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
df[['Height','Weight']]=scalar.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="800" height="557" alt="image" src="https://github.com/user-attachments/assets/9aa4ba68-7b72-4f21-9344-1113f653ad72" />

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="858" height="628" alt="image" src="https://github.com/user-attachments/assets/922f1a67-2180-4329-a2a4-c69045a680da" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="902" height="632" alt="image" src="https://github.com/user-attachments/assets/8e861499-748f-40cd-bbbe-0e703551656f" />

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
<img width="885" height="452" alt="image" src="https://github.com/user-attachments/assets/73c7af30-bf73-4bf0-a0fb-cb376f8f1002" />

```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="910" height="375" alt="image" src="https://github.com/user-attachments/assets/ab723517-c7e7-476c-ac3e-14fea9d87331" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="602" height="135" alt="image" src="https://github.com/user-attachments/assets/bc9ca472-838b-4e8c-9fb1-1ad1797bfff8" />
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
```
<img width="740" height="81" alt="image" src="https://github.com/user-attachments/assets/52eb088a-bc22-49b1-99a3-b8110f76cf16" />
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
<img width="748" height="102" alt="image" src="https://github.com/user-attachments/assets/08f76f20-29d9-4d94-b7dd-8c85b97456ea" />


# RESULT:
Feature scaling and feature selection process has been successfullyperformed on the data set.
    
