import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model as lm
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import train_test_split as tts

#############################LOAD THE DATA#####################################
movies = pd.read_csv('datasets/moviesCleanedImputatedLog.csv',sep='_',encoding = "ISO-8859-1")
print(movies.head())

##############################PREPARE DATA#####################################
#Data is prepared for the analysis
movAtt = movies[['budget','revenue','profit','genres','keywords','production_companies','runtime','vote_count','vote_average','popularity']]
columnNames = list(movAtt)
movTarget = np.asarray(movies.isAwarded)

#Data should be normalized
movAtt = (movAtt-movAtt.mean())/movAtt.std()
movAttMat = movAtt.as_matrix()

############################TRAIN TEST SPLIT###################################
xTr,xTest,yTr,yTest = tts(movAttMat, movTarget, test_size=0.1, random_state=42)

#Generate indices for cross validation
xTrain,xCv,yTrain,yCv = tts(xTr,yTr,test_size=0.23,random_state = 42)

###########################RANDOM FOREST#######################################
#Train of the random forest, grid search on the number of trees
maxTrees = 200
errModel = []
for i in range(1,maxTrees):
    rf = RandomForestClassifier(n_estimators = i)
    rf.fit(xTrain,yTrain)
    
    #Test validation performance
    confMatrix = cm(yCv,rf.predict(xCv))
    errModel.append(1-(confMatrix[0,0]+confMatrix[1,1])/sum(sum(confMatrix)))
    
plt.figure(1)
plt.plot(errModel)
plt.title('Error vs. number of trees')

#Train on the best parameters
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(xTrain,yTrain)

#Importance of the features
importantFeatRF = np.asarray(rf.feature_importances_)[np.argsort(np.abs(rf.feature_importances_))[::-1]]
importantNamesRF = np.asarray(columnNames)[np.argsort(np.abs(rf.feature_importances_))[::-1]]

print("\nRANDOM FOREST IMPORTANT FEATURES:")
for i in range(len(importantFeatRF)):
    print(" - Variable ",importantNamesRF[i]," has a weight: ",importantFeatRF[i])

#Train 
confMatRFTr = cm(yTrain,rf.predict(xTrain))
trainErrRF = 1-(confMatRFTr[0,0]+confMatRFTr[1,1])/sum(sum(confMatRFTr))

#Cross validation
confMatRFCv = cm(yCv,rf.predict(xCv))
cvErrRF = 1-(confMatRFCv[0,0]+confMatRFCv[1,1])/sum(sum(confMatRFCv))

#Test 
confMatRFTest = cm(yTest,rf.predict(xTest))
testErrRF = 1-(confMatRFTest[0,0]+confMatRFTest[1,1])/sum(sum(confMatRFTest))

print("\nRANDOM FOREST ERROR:")
print("Train error:     ",trainErrRF)
print("Cross val error: ",cvErrRF)
print("Test error:      ",testErrRF)
    
###########################LOGISTIC REGRESSION#################################
#Train of logistic regression
log = lm.LogisticRegression()
log.fit(xTrain,yTrain)

#Importance of the features
importantFeatLR = log.coef_[0][np.argsort(np.abs(log.coef_[0]))[::-1]]
importantNamesLR = np.asarray(columnNames)[np.argsort(np.abs(log.coef_[0]))[::-1]]

print("\nLOGISTIC REGRESSION IMPORTANT FEATURES:")
for i in range(len(importantFeatLR)):
    print(" - Variable ",importantNamesLR[i]," has a weight: ",importantFeatLR[i])
    
#Train
confMatLGTr = cm(yTrain,log.predict(xTrain))
trainErrLG = 1-(confMatLGTr[0,0]+confMatLGTr[1,1])/sum(sum(confMatLGTr))

#Cross validation
confMatLGCv = cm(yCv,log.predict(xCv))
cvErrLG = 1-(confMatLGCv[0,0]+confMatLGCv[1,1])/sum(sum(confMatLGCv))

#Test
confMatLGTest = cm(yTest,log.predict(xTest))
testErrLG = 1-(confMatLGTest[0,0]+confMatLGTest[1,1])/sum(sum(confMatLGTest))

print("\nLOGISTIC REGRESSION ERROR:")
print("Train error:     ",trainErrLG)
print("Cross val error: ",cvErrLG)
print("Test error:      ",testErrLG)