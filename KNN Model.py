# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 10:13:11 2022

@author: Dayanand
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

# increase the display size
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# change directory
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\Job-a-thon")

# loading file train & test datasets.Calling test file to prediction dataset

rawDf=pd.read_csv("train_0OECtn8.csv")
predictionDf=pd.read_csv("test_1zqHu22.csv")

rawDf.shape
predictionDf.shape

# we have one more column in prediction then raw.Let us see columns
rawDf.columns
predictionDf.columns
# engagement_score column is more in prediction dataset.Let us add this column
predictionDf["engagement_score"]=0
predictionDf.shape

# Let us divide the rawDf into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)

trainDf.shape
testDf.shape

# Let us source column in train,test & prediction

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

# Let us combine all three datasets for data processing
fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullDf.shape

# let us drop identifier columns which are not of use

fullDf.columns
fullDf.drop(["row_id","user_id","category_id","video_id"],axis=1,inplace=True)
fullDf.shape

# let us check NULL values
fullDf.isna().sum() # No Null values

#Bivariate Analysis Continuous Variables:Scatter plot

corrDf=fullDf[fullDf["Source"]=="Train"].corr() #inference always shold be from Train data 
corrDf.head

sns.heatmap(corrDf,
            xticklabels=corrDf.columns,
            yticklabels=corrDf.columns,
            cmap='YlOrBr')

# Bivariate Analysis Categorical Variables:Boxplot
sns.boxplot(y=trainDf["engagement_score"],x=trainDf["gender"])
# Male is more engagement_score than female
sns.boxplot(y=trainDf["engagement_score"],x=trainDf["profession"])
# other and working_professional has almost same levele of engagement_score

# dummy variable creation
fullDf2=pd.get_dummies(fullDf,drop_first=False)
fullDf2.shape

############################
# Divide the data into Train and Test
############################
# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column

# Step 1: Divide into Train and Testest

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy() 
predictDf=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()

########################
# Sampling into X and Y
########################

# Divide each dataset into Indep Vars and Dep var

depVar="engagement_score"
trainX=trainDf.drop([depVar],axis=1)
trainY=trainDf[depVar]

testX=testDf.drop([depVar],axis=1)
testY=testDf[depVar]

predictX=predictDf.drop([depVar],axis=1)

trainX.shape
trainY.shape
testX.shape
testY.shape
predictX.shape


# Model Building Using KNNwith default parameter
from sklearn.neighbors import KNeighborsRegressor
M1_KNN=KNeighborsRegressor().fit(trainX,trainY)

# Prediction om TestSet
Test_Predict1=M1_KNN.predict(testX)

# Import r2_score
from sklearn.metrics import r2_score
R2_Score1=r2_score(testY,Test_Predict1) #0.2482

# Let us standarized the variables

trainXCopy=trainX.copy()
testXCopy=testX.copy()
predictXCopy=predictX.copy()

from sklearn.preprocessing import StandardScaler

train_Sampling=StandardScaler().fit(trainXCopy)
trainXStd=train_Sampling.transform(trainXCopy)
testXStd=train_Sampling.transform(testXCopy)
predictXStd=train_Sampling.transform(predictXCopy)

trainXStd=pd.DataFrame(trainXStd,columns=trainXCopy.columns)
testXStd=pd.DataFrame(testXStd,columns=testXCopy.columns)
predictXStd=pd.DataFrame(predictXStd,columns=predictXCopy.columns)

# Model Building Using KNNwith default parameter on standarized datasets

from sklearn.neighbors import KNeighborsRegressor
M1_KNN1=KNeighborsRegressor().fit(trainXStd,trainY)

# Prediction om TestSet
Test_Predict2=M1_KNN1.predict(testXStd)

# Import r2_score
from sklearn.metrics import r2_score
R2_Score2=r2_score(testY,Test_Predict1)
R2_Score2 #0.248

#GridSearch CV on standardized data sets

from sklearn.model_selection import GridSearchCV

myNN = range(1,14,2) # list(range(1,14,2))
myP = range(1,4,1) # list(range(1,4,1)) . This "p" is "h" in minkowski formula
my_param_grid = {'n_neighbors': myNN, 'p': myP} # param_grid is a dictionary 

Grid_Search_Model = GridSearchCV(estimator = KNeighborsRegressor(), 
                     param_grid=my_param_grid,  
                     scoring='r2', 
                     cv=5, n_jobs = -1).fit(trainXStd, trainY)
# Other scoring parameters are available here: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

Grid_Search_Df = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)

# Model Building Using KNN with tuning parametrs


M1_KNN2=KNeighborsRegressor(n_neighbors=13,p=1).fit(trainXStd,trainY)

# Prediction om TestSet
Test_Predict3=M1_KNN2.predict(testXStd)

# Import r2_score

R2_Score3=r2_score(testY,Test_Predict3)
R2_Score2 #0.248



# Prediction on PredictionDataSets

SampleSubmissionKNN=pd.DataFrame()
SampleSubmissionKNN["row_id"]=predictionDf["row_id"]
SampleSubmissionKNN["engagement_score"]=M1_KNN2.predict(predictXStd)
SampleSubmissionKNN.to_csv("SampleSubmissionKNN.csv",index=False)
