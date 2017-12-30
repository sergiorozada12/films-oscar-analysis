import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as knn

#############################FUNCTIONS#########################################
#This function trains a KNN classifier given the proper features
def trainKNN(target,ecoMarker,socialMarker,contentMarker):
    #Creation of dataframe and normalization (To make distances comparable)
    df = pd.concat([ecoMarker,socialMarker,contentMarker],axis=1)
    df = (df-df.min())/(df.max()-df.min())
    target = (target - target.min())/(target.max()-target.min())
    
    #Fit of the classifier and return
    classifier=knn(n_neighbors=5)
    classifier.fit(df,target)
    print("\nKNN score : ",classifier.score(df,target))
    return classifier

###############################################################################
#This function predicts missing values given knn neighbours
def predictKNN(targetComplete,ecoComplete,socComplete,contComplete,ecoMarker,socialMarker,contentMarker,classifier):
    #Creation of the dataframe given the features and normalization
    ecoMarker = (ecoMarker-ecoComplete.min())/(ecoComplete.max()-ecoComplete.min())
    socialMarker = (socialMarker-socComplete.min())/(socComplete.max()-socComplete.min())
    contentMarker = (contentMarker-contComplete.min())/(contComplete.max()-contComplete.min())
    df = pd.concat([ecoMarker,socialMarker,contentMarker],axis=1)
    
    #Prediction, denormalization and return
    target = classifier.predict(df)
    return target*(targetComplete.max()-targetComplete.min())+targetComplete.min()


############################LOAD DATA##########################################
#Loading of the engineered prepared dataset
movies=pd.read_csv('datasets/moviesEngineered.csv',sep='_',encoding = "ISO-8859-1")

##########################INITIAL SCREENING####################################

#Look for missing values which are explicit in the dataset. No data is missing
print("MISSING DATA IN THE DATASET BEFORE IMPUTATION")
print(movies.isnull().sum())

#Check wheter the data is balanced or not(Awarded vs Not awarded)
print("\nNUMBER OF FILMS AWARDED AND NOT BEFORE IMPUTATION")
print(movies.isAwarded.value_counts())

###############################CASE DELETION###################################

#If we take a close look to the dataset we see that lot of data is not missed directly(NaN) but
#set to zero. This creates a huge impact on the statistics of the data, fields like budget
# or revenue are not likely to be set zero. Other data like genres or keywords
#can also be found empty.

#The following films are disregarded: 
#   - 1)No budget AND no revenue information -> No information about economic dimension
#   - 2)Less than 20 votes -> No information about social dimension
#Data not fullfilling 1 AND 2 are not considered (very little information)
selected=movies[((movies.revenue>0)|(movies.budget>0))&(movies.vote_count>20)].reset_index(drop=True)

###############################DATASET I: PURE#################################
#Dataset is split in pure cases (full info), data only lacking Budget information
#and data only lacking revenue information.
selectedComplete=selected[(selected.revenue>0)&(selected.budget>0)&(selected.genres>0)&(selected.genres>0)&(selected.keywords>0)&(selected.production_companies>0)].reset_index(drop=True)
selectedZeroBudget=selected[(selected.budget==0)].reset_index(drop=True)
selectedZeroRevenue=selected[(selected.revenue==0)].reset_index(drop=True)
selectedZeroGenres=selected[(selected.genres==0)].reset_index(drop=True)
selectedZeroKeywords=selected[(selected.keywords==0)].reset_index(drop=True)
selectedZeroProdComp=selected[(selected.production_companies==0)].reset_index(drop=True)

#We see that the number of budget missing values is low (3%), however zero revenue
#is higher (12%)
print("\nNUMBER OF CASES WITH ZERO REVENUE, BUDGET AND COMPLETE")
print("Complete cases: ",selectedComplete.shape[0])
print("Zero revenue: ",selectedZeroRevenue.shape[0])
print("Zero budget: ",selectedZeroBudget.shape[0])
print("Zero genres: ",selectedZeroGenres.shape[0])
print("Zero keywords: ",selectedZeroKeywords.shape[0])
print("Zero production companies: ",selectedZeroProdComp.shape[0])

#We will save the dataset which is "pure" to further use. (Comparison between
#imputation methods) Perhaps the analysis can be fully performed on this dataset
#but some imputation method could improve the results (Eg: Missing data can be MNAR
#and related with the popularity of the film)
selectedComplete.to_csv('datasets/moviesCleanedPure.csv',sep='_',index=False)

###########################MEDIAN IMPUTATION####################################
#Content features is median imputated: genres, keywords and production_companies (zeros affect)
selected.genres[selected.genres==0]=selected.genres.median()
selected.production_companies[selected.production_companies==0]=selected.production_companies.median()
selected.keywords[selected.keywords==0]=selected.keywords.median()

##########################KNN IMPUTATION#####################################
#KNN: We are going to create a dataset with missing values being similar to their
#neighbors (budget-revenue-vote count->economic and social impact). It should be more robust
#DRAWBACK: We are creating points based on similitudes in a large number of rows that
#may affect the quality of the analysis creating artificial interdependences and correlations.
budgetClassifier = trainKNN(selectedComplete.budget,selectedComplete.revenue,selectedComplete.vote_count,selectedComplete.keywords)
selectedZeroBudget.budget = predictKNN(selectedComplete.budget,selectedComplete.revenue,selectedComplete.vote_count,selectedComplete.genres,selectedZeroBudget.revenue,selectedZeroBudget.vote_count,selectedZeroBudget.genres,budgetClassifier)

revenueClassifier = trainKNN(selectedComplete.revenue,selectedComplete.budget,selectedComplete.vote_count,selectedComplete.keywords)
selectedZeroRevenue.revenue = predictKNN(selectedComplete.revenue,selectedComplete.budget,selectedComplete.vote_count,selectedComplete.genres,selectedZeroRevenue.budget,selectedZeroRevenue.vote_count,selectedZeroRevenue.genres,revenueClassifier)

#Finally we create the dataset with the "pure" data, plus zero budget and zero revenue
#imputed by knn
selectedKNN=pd.concat([selectedComplete,selectedZeroRevenue,selectedZeroBudget])
selectedKNN=selectedKNN.reset_index(drop=True)

#Recalculation of profit
selectedKNN.profit = selectedKNN.revenue-selectedKNN.budget
selectedKNN.to_csv('datasets/moviesCleanedImputated.csv',sep='_',index=False)

############################FINAL SCREENING####################################
#We look at some statistical values of our datasets
print("\nSTATISTICS OF DATASETS WITH PURE VALUES WITHOUT IMPUTATIONS")
print(selectedComplete.describe())

print("\nSTATISTICS OF DATASETS WITH KNN IMPUTATION")
print(selectedKNN.describe())
