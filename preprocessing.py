import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
X=df.iloc[:,0:16]
y = pd.DataFrame(df.iloc[:,16])
X_encoded = X.copy()
y_encoded = y.copy()
#unimos X_encoded y y_encoded para hacer la codificacion de las variables categoricas
X_encoded = pd.concat([X_encoded, y_encoded], axis=1)
X_encoded

#Redondeamos los valores de la variable NCP a 0 decimales y si hay un valor mayor a 3 lo redondeamos a 3
X_encoded['NCP'] = X_encoded['NCP'].round(0).astype(int)
X_encoded.loc[X_encoded['NCP'] > 3, 'NCP'] = 3
#Redondeamos los valores de las variable FCVC y TUE a 0 decimales y convertimos a entero
X_encoded['FCVC'] = X_encoded['FCVC'].round(0).astype(int)
X_encoded['TUE'] = X_encoded['TUE'].round(0).astype(int)
X_encoded['CH2O'] = X_encoded['CH2O'].round(0).astype(int)
X_encoded['FAF'] = X_encoded['FAF'].round(0).astype(int)
#Convertimos las variables Gender, SMOKE,SCC,FAVC y family_history_with_overweight a 0 y 1
X_encoded['SCC'] = X_encoded['SCC'].map({'yes': 1, 'no': 0})
X_encoded['FAVC'] = X_encoded['FAVC'].map({'yes': 1, 'no': 0})
#usamos ordinal encoding para las variables CALC, CAEC y NObeyesdad
X_encoded['CALC'] = X_encoded['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
X_encoded['CAEC'] = X_encoded['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
X_encoded['NObeyesdad'] = X_encoded['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6})
#usamos one hot encoding para la variable MTRANS, Gender, SMOKE y family_history_with_overweight
dummies1 = pd.get_dummies(X_encoded[['MTRANS']],  prefix='MTRANS').astype(int)
dummies2 = pd.get_dummies(X_encoded[['Gender']],  prefix='Gender').astype(int)
dummies3 = pd.get_dummies(X_encoded[['SMOKE']],  prefix='SMOKE').astype(int)
dummies4 = pd.get_dummies(X_encoded[['family_history_with_overweight']],  prefix='family_history_with_overweight').astype(int)
#Concatenamos las variables dummies
dummies = pd.concat([dummies1, dummies2, dummies3, dummies4], axis=1)
#hacemos drop de las columnas MTRANS, Gender, SMOKE y family_history_with_overweight
X_encoded = X_encoded.drop(['MTRANS','Gender','SMOKE','family_history_with_overweight'],axis=1)
X_encoded = pd.concat([X_encoded, dummies], axis=1)

#Unimos las variables con la variable objetivo
df_obsesity = X_encoded.copy()
#Separamos las variables numericas en un dataset
df_obsesity_num = X_encoded.copy()
#Drop de las culumnas height, weight y NObeyesdad
X_encoded = X_encoded.drop(['Height','Weight','NObeyesdad'],axis=1)

#Concat de X y y
Finaldf = pd.concat([X_encoded, y_encoded], axis=1)
# Exportamos a csv
Finaldf.to_csv('Preprocessed_DataSet.csv', index=False)
