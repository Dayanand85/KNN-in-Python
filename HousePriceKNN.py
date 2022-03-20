# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:37:04 2022

@author: Dayanand
"""

### loading library
import os
import pandas as pd
import seaborn as sns
import numpy as np

### setting display size
pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# changing directory
os.chdir("C:/Users/Dayanand/Desktop/DataScience/dsp1/Kaggle/DataSets")

# loading library
rawData=pd.read_csv("HousePrices.train.csv")
predictionData=pd.read_csv("HousePrices.test.csv")
rawData.shape
predictionData.shape
rawData.columns
predictionData.columns

# Add SalePrice to predictionData
predictionData["SalePrice"]=0

# Divide rawData into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawData,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# Add Source column in all three data sets
trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionData["Source"]="Prediction"

# Add all three data sets
fullDf=pd.concat([trainDf,testDf,predictionData],axis=0)
fullDf.shape

# Remove identifier column
fullDf.drop(["Id"],axis=1,inplace=True)
fullDf.columns
# Null Values check
fullDf.isna().sum()

# Imputing Missing Values

for i in fullDf.columns:
    if i!="SalePrice" and i!="Source":
        if fullDf[i].dtypes=="object":
            #print(fullDf[i])
            tempMode=fullDf.loc[fullDf["Source"]=="Train",i].mode()[0]
            fullDf[i].fillna(tempMode,inplace=True)
        else:
            tempMedian=fullDf.loc[fullDf["Source"]=="Train",i].median()
            fullDf[i].fillna(tempMedian,inplace=True)
        

fullDf.isna().sum()

## let us create dummy variables
fullDf2=pd.get_dummies(fullDf)
fullDf2.shape

### let us divide the data sets
trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
trainDf.shape
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
testDf.shape
predictionDf=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
predictionDf.shape

trainX=trainDf.drop(["SalePrice"],axis=1)
trainX.shape
trainY=trainDf["SalePrice"]
trainY.shape
testX=testDf.drop(["SalePrice"],axis=1)
testX.shape
testY=testDf["SalePrice"]
testY.shape
predictionX=predictionDf.drop(["SalePrice"],axis=1)
predictionX.shape

### Standardized data
from sklearn.preprocessing import StandardScaler
trainStandard= StandardScaler().fit(trainX)
trainXStd=trainStandard.transform(trainX)
trainXStd=pd.DataFrame(trainXStd,columns=trainX.columns)
testXStd=trainStandard.transform(testX)
testXStd=pd.DataFrame(testXStd,columns=testX.columns)
predictionXStd=trainStandard.transform(predictionX)

####Model Building
from sklearn.neighbors import KNeighborsRegressor
KNN1=KNeighborsRegressor(n_neighbors=5).fit(trainXStd,trainY)
Test_Predict=KNN1.predict(testXStd)

# RMSE,MAPE
np.sqrt((np.mean(testY-Test_Predict)**2))
### 1265
(np.mean(abs((testY-Test_Predict)/testY)))*100
# 16%

Prediction_Predict=KNN1.predict(predictionXStd)
SubmissionFile=pd.concat([pd.DataFrame(predictionData["Id"]),pd.DataFrame(Prediction_Predict)],axis=1)
SubmissionFile.columns=["Id","SalePrice"]
SubmissionFile.set_index("Id",inplace=True)
SubmissionFile.to_csv("SubmissionFile.csv")


#####
## Model Building GradintBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
GBR_Model=GradientBoostingRegressor().fit(trainXStd,trainY)
Var_Imp=pd.DataFrame(GBR_Model.feature_importances_)
Var_Imp=pd.concat([Var_Imp,pd.DataFrame(trainX.columns)],axis=1)
Var_Imp.columns=["Gini_Value","Column_Name"]
Test_Predict=GBR_Model.predict(testXStd)

#### MAPE
(np.mean(abs((testY-Test_Predict)/testY)))*100
### 10%

# GridSearchCV
from sklearn.model_selection import GridSearchCV

nEstimator=[15,25,45,75,100]
MinLeaf=[100,200,300,400,500]
MaxFeatures=[5,8,12,16,25]
myParamGrid={"n_estimator":nEstimator,
             "min_samples_leaf":MinLeaf,
             "max_features":MaxFeatures}
GBRFGrid=GridSearchCV(estimator=GradientBoostingRegressor(random_state=2410),
                      param_grid=myParamGrid,scoring=None,cv=4).fit(trainXStd,trainY)


### Hyper parameter
GBR_Model2=GradientBoostingRegressor(random_state=2410,n_estimators=75,max_features=10,
                                     min_samples_leaf=50).fit(trainXStd,trainY)

Test2_Predict=GBR_Model2.predict(testXStd)

# MAPE
(np.mean(abs(((testY-Test2_Predict)/testY))))*100
### 12%

### AdaBoost model Building
from sklearn.ensemble import AdaBoostRegressor
ABR_Model=AdaBoostRegressor(random_state=2410).fit(trainXStd,trainY)
Test_Predict1=ABR_Model.predict(testXStd)

### MAPE
(np.mean(abs((testY-Test_Predict1)/testY)))*100
## 17.54

## Hyper Parameter
ABR_Model1=AdaBoostRegressor(random_state=2410,n_estimators=75,learning_rate=1
                             ).fit(trainXStd,trainY)

Test_Predict2=ABR_Model1.predict(testXStd)

### MAPE
(np.mean(abs((testY-Test_Predict2)/testY)))*100
### 