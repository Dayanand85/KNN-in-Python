# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:36:08 2022

@author: Dayanand
"""

# loading library

import os
import pandas as pd
import numpy as np
import seaborn as sns

# setting display size
pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# changing directory
os.curdir
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\DataSets")

# loading file
rawData=pd.read_csv("cancerdata.csv")

# NULL values check
rawData.isna().sum() # No null values

# chacking event rate

rawData.columns
rawData.describe()
rawData["diagnosis"].value_counts() # B-majority class

# drop identifier columns

rawData.drop(["id"],axis=1,inplace=True)
rawData.shape

# changing depe. var column into 0&1 class

rawData["diagnosis"]=np.where(rawData["diagnosis"]=="B",0,1)
rawData["diagnosis"].value_counts()

# sampling data into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawData,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# checking trainDf event rate of dependent variable
trainDf["diagnosis"].value_counts()# 0 class is majority class

# dividing dep & indep variables
trainX=trainDf.drop(["diagnosis"],axis=1)
trainY=trainDf["diagnosis"]
testX=testDf.drop(["diagnosis"],axis=1)
testY=testDf["diagnosis"]
trainX.shape
testX.shape

# standarsized datasets

from sklearn.preprocessing import StandardScaler
train_samp=StandardScaler().fit(trainX)
trainStd=train_samp.transform(trainX)
testStd=train_samp.transform(testX)

# Adding column names
trainX_Std=pd.DataFrame(trainStd,columns=trainX.columns)
testX_Std=pd.DataFrame(testStd,columns=testX.columns)

# Model building
from sklearn.neighbors import KNeighborsClassifier

M1=KNeighborsClassifier().fit(trainX_Std,trainY)

# Model prediction on testData sets

test_pred=M1.predict(testX_Std)

# Creating a dataframe to store testpred
test_prob_df=pd.DataFrame()
test_prob_df["class"]=test_pred


# Model Evaluation
from sklearn.metrics import confusion_matrix,classification_report

conf_mat=confusion_matrix(testY,test_pred)
conf_mat

print(classification_report(testY,test_pred))

# crating probabilities of class on train as well as on test data

# Test Data sets
# Test_Pred=pd.DataFrame(M1.predict_proba(testX_Std))
# test_prob_df=pd.concat([test_prob_df,Test_Pred],axis=1)

# Train Data Sets
Train_Pred=M1.predict(trainX_Std)
Train_Prob=pd.DataFrame()
Train_Prob["class"]=Train_Pred
Train_Pred_Prob=pd.DataFrame(M1.predict_proba(trainX_Std))
Train_Prob_Df=pd.concat([Train_Prob,Train_Pred_Prob],axis=1)

Train_Prob_Df["Pred_Prob"]=1-Train_Prob_Df.iloc[:,1]
Pred_Prob=Train_Prob_Df["Pred_Prob"]

# ROC curve
from sklearn.metrics import roc_curve,auc

# Error-getting only 7 values for tpr & fpr

fpr,tpr,cutoff=roc_curve(trainY,Pred_Prob)

# cut_off table creation
cut_off_table=pd.DataFrame()
cut_off_table["fpr"]=fpr
cut_off_table["tpr"]=tpr
cut_off_table["cut_off"]=cutoff

# Area Under Curve
auc(fpr,tpr)# 99.5%

# GridSearch CV
from sklearn.model_selection import GridSearchCV
myNN=range(1,14,2)
myP=range(1,5,1)
myparam={"n_neighbors":myNN,"p":myP}#My param is a ditionary

grid_search_model=GridSearchCV(estimator=KNeighborsClassifier(),
                           param_grid=myparam,
                           scoring="accuracy",
                           cv=5,n_jobs=-1).fit(trainX_Std,trainY)
gird_search_df=pd.DataFrame.from_dict(grid_search_model.cv_results_)

# Data transformation
col_to_consider=["perimeter_mean","area_mean","perimeter_worst","area_worst"]
sns.pairplot(trainX[col_to_consider])

# let us do log or sqrt transform 
trainX_Copy=trainX.copy()
trainX_Copy["perimeter_mean"]=np.sqrt(np.where(trainX_Copy["perimeter_mean"]==0,1,trainX_Copy["perimeter_mean"]))
trainX_Copy["area_mean"]=np.log(np.where(trainX_Copy["area_mean"]==0,1,trainX_Copy["area_mean"]))
trainX_Copy["perimeter_worst"]=np.sqrt(np.where(trainX_Copy["perimeter_worst"]==0,1,trainX_Copy["perimeter_worst"]))
trainX_Copy["area_worst"]=np.log(np.where(trainX_Copy["area_worst"]==0,1,trainX_Copy["area_worst"]))

testX_Copy=testX.copy()
testX_Copy["perimeter_mean"]=np.sqrt(np.where(testX_Copy["perimeter_mean"]==0,1,testX_Copy["perimeter_mean"]))
testX_Copy["area_mean"]=np.log(np.where(testX_Copy["area_mean"]==0,1,testX_Copy["area_mean"]))
testX_Copy["perimeter_worst"]=np.sqrt(np.where(testX_Copy["perimeter_worst"]==0,1,testX_Copy["perimeter_worst"]))
testX_Copy["area_worst"]=np.log(np.where(testX_Copy["area_worst"]==0,1,testX_Copy["area_worst"]))

# histogram using pairplot
sns.pairplot(trainX_Copy[col_to_consider])

# standardized the datasets

trainX_Copy_Sampling=StandardScaler().fit(trainX_Copy)
trainX_Copy_Std=trainX_Copy_Sampling.transform(trainX_Copy)
testX_Copy_Std=trainX_Copy_Sampling.transform(testX_Copy)

# Add the column names to the datasets
trainX_Std=pd.DataFrame(trainX_Copy_Std,columns=trainX.columns)
testX_Std=pd.DataFrame(testX_Copy_Std,columns=testX.columns)

sns.pairplot(trainX_Std[col_to_consider])

# Model Building
M2=KNeighborsClassifier(n_neighbors=5,p=1).fit(trainX_Std,trainY)

# Predict on Test DataSets
test_pred=M2.predict(testX_Std)

# Model Evaluation
conf_mat=confusion_matrix(testY,test_pred)
conf_mat
print(classification_report(testY,test_pred))
# accuracy-93%