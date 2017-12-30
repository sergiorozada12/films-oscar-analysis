import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################LOAD DATA########################################
#Load the file and print out some basic statistics
movies = pd.read_csv('datasets/moviesCleanedImputated.csv',sep='_',encoding = "ISO-8859-1")

##############################SOME CHANGES#####################################
#Date is set as index
movies.release_date = pd.to_datetime(movies.release_date)
movies = movies.sort_values(by='release_date')
movies.index = movies['release_date']

#Profit is converted into percentage
movies.profit = movies.profit.divide(movies.budget)*100

#We get rid of some outliers (Most profitable movie in history 20000%)
print("\nOutliers eliminated: ",movies.title[movies.profit > 20000])
movies.profit[movies.profit > 20000] = 0

###############################FIRS SCREENING##################################
#Basic statistics of total movies
print(movies.head())
print("\nBASIC STATISTICS:\n")
print(movies.describe())

#Basic statistics of awarded films
print("\nAWARDED BASIC STATISTICS:\n")
print(movies[movies.isAwarded==True].describe())

#Basic statistics of non-awarded movies
print("\nNON AWARDED BASIC STATISTICS:\n")
print(movies[movies.isAwarded==False].describe())

#################################SPLIT OF DATASET##############################
#Dataset is divided into target, attributes, movies and dates
movieNames = movies.title
movieTarget = movies.isAwarded
movieAttributes = movies[['budget','revenue','profit','genres','keywords','production_companies','runtime','vote_count','vote_average','popularity']]
attributes = list(movieAttributes)

#############################ECONOMIC PATTERNS#################################
#Temporal patterns can be addresed in economic variables
#Data is resample by year
attYear = movieAttributes.resample('A').mean()

#Outliers are eliminated
attYear = attYear.fillna(0)

#Evolution of budget and revenue over time (looking at max values)
plt.figure(1,figsize=(10,10))
plt.title('Budget and Revenue (Corrected for inflation')
plt.plot(attYear.index.year,attYear.budget,c='r')
plt.plot(attYear.index.year,attYear.revenue,c='g')
plt.legend()

#Evolution of the profit by percentage
plt.figure(2,figsize=(10,10))
plt.title('Profit (%) (Corrected for inflation')
plt.plot(attYear.index.year,attYear.profit)

############################SEASONAL PATTERNS##################################
#Seasonal patterns can be addresed resampling the data by month
attMonthAw = list(movies[movies.isAwarded==True].index.month)
attMonthAw = sorted(attMonthAw)
attMonthNaw = list(movies[movies.isAwarded==False].index.month)
attMonthNaw = sorted(attMonthNaw)
movAw = {x:attMonthAw.count(x) for x in attMonthAw}
movNaw = {x:attMonthNaw.count(x) for x in attMonthNaw}

#Plot of the movies awarded and not awarded depending on the months
plt.figure(3,figsize=(10,10))
plt.title('Seasonal pattern')
bar_width = 0.4
a = plt.bar(list(movAw.keys()),list(movAw.values()),bar_width,color='g')
b = plt.bar(np.array(list(movNaw.keys()))+bar_width,list(movNaw.values()),bar_width,color='r')
plt.ylabel('movies')
plt.xlabel('month')
plt.legend((a,b),('awarded','not awarded'))

#######################CORRELATIONS AND LOG TRANSFORM##########################
##Applying log transformations to the skewed the data and compare the difference
movieLog = movieAttributes.copy()
movieLog[['budget','revenue','vote_count','popularity']] = movieAttributes[['budget','revenue','vote_count','popularity']].apply(lambda x: np.log(x))

#Let's analyse relations between variables (without and log transformed)
#Covariance matrices
covMatrix = movieAttributes.cov()
covMatrixLog = movieLog.cov()
print("\nCOVARIANCE MATRIX:\n",covMatrix)
print("\nCOVARIANCE MATRIX LOG:\n",covMatrixLog)

#Pearson correlation matrices
pearsonMatrix = movieAttributes.corr('pearson')
pearsonMatrixLog = movieLog.corr('pearson')
print("\nPEARSON MATRIX:\n",pearsonMatrix)
print("\nPEARSON MATRIX LOG:\n",pearsonMatrixLog)

#Spearman correlation matrices
spearmanMatrix = movieAttributes.corr('spearman')
spearmanMatrixLog = movieLog.corr('spearman')
print("\nSPEARMAN MATRIX:\n",spearmanMatrix)
print("\nSPEARMAN MATRIX LOG:\n",spearmanMatrixLog)

##############################HEATMAPS#########################################
#Pearson correlation of non-transformed data
plt.figure(4,figsize=(10,10))
plt.title('Pearson correlation')
plt.pcolor(pearsonMatrix)
plt.gca().invert_yaxis()
plt.yticks(range(len(attributes)),attributes)
plt.xticks(range(len(attributes)),attributes)
plt.colorbar()

#Pearson correlation of transformed data
plt.figure(5,figsize=(10,10))
plt.title('Pearson correlation LOG')
plt.pcolor(pearsonMatrixLog)
plt.gca().invert_yaxis()
plt.yticks(range(len(attributes)),attributes)
plt.xticks(range(len(attributes)),attributes)
plt.colorbar()

#Spearman correlation of non-transformed data
plt.figure(6,figsize=(10,10))
plt.title('Spearman correlation')
plt.pcolor(spearmanMatrix)
plt.gca().invert_yaxis()
plt.yticks(range(len(attributes)),attributes)
plt.xticks(range(len(attributes)),attributes)
plt.colorbar()

#Spearman correlation of transformed data
plt.figure(7,figsize=(10,10))
plt.title('Spearman correlation LOG')
plt.pcolor(spearmanMatrixLog)
plt.gca().invert_yaxis()
plt.yticks(range(len(attributes)),attributes)
plt.xticks(range(len(attributes)),attributes)
plt.colorbar()

##########################HISTOGRAMS BIMODAL DATA##############################
#Some distributions are bimodal, perhaps awarded-nonawarded shows different distributions
#Plot histograms of the bimodal variables genres and producer_companies (isAwarded the key?)
plt.figure(8,figsize=(10,10))
movieAttributes.genres[movieTarget==True].plot.kde(color='g')
movieAttributes.genres[movieTarget==False].plot.kde(color='r')

plt.figure(9,figsize=(10,10))
movieAttributes.production_companies[movieTarget==True].plot.kde(color='g')
movieAttributes.production_companies[movieTarget==False].plot.kde(color='r')

#######################3D SCATTER BY DIMENSION#################################
#Scatter social,economic and content normal data and log to see variables in space 
#by theme and log-normalized. See relations inside dimension
colorMap = []
for target in movieTarget:
    if target==0:
        colorMap.append('g')
    else:
        colorMap.append('r')
        
economicMovies = movieAttributes[['revenue','profit','budget']].as_matrix()
socialMovies = movieAttributes[['vote_count','vote_average','popularity']].as_matrix()
contentMovies = movieAttributes[['keywords','genres','runtime']].as_matrix()

economicMoviesLog = movieLog[['revenue','profit','budget']].as_matrix()
socialMoviesLog = movieLog[['vote_count','vote_average','popularity']].as_matrix()
contentMoviesLog = movieLog[['keywords','genres','runtime']].as_matrix()

#Economic dimension
fig = plt.figure(10)
ax = fig.add_subplot(111, projection='3d',title='Economic variables')
ax.scatter(economicMovies[:,0],economicMovies[:,1],economicMovies[:,2],c=colorMap)
ax.set_xlabel('revenue')
ax.set_ylabel('profit')
ax.set_zlabel('budget')

#Log economic dimension
fig = plt.figure(11)
ax = fig.add_subplot(111, projection='3d',title="Economic variables LOG")
ax.scatter(economicMoviesLog[:,0],economicMoviesLog[:,1],economicMoviesLog[:,2],c=colorMap)
ax.set_xlabel('revenue')
ax.set_ylabel('profit')
ax.set_zlabel('budget')

#Social dimension
fig = plt.figure(12)
ax = fig.add_subplot(111, projection='3d',title="Social variables")
ax.scatter(socialMovies[:,0],socialMovies[:,1],socialMovies[:,2],c=colorMap)
ax.set_xlabel('vote_count')
ax.set_ylabel('vote_average')
ax.set_zlabel('popularity')

#Log social dimension
fig = plt.figure(13)
ax = fig.add_subplot(111, projection='3d',title="Social variables LOG")
ax.scatter(socialMoviesLog[:,0],socialMoviesLog[:,1],socialMoviesLog[:,2],c=colorMap)
ax.set_xlabel('vote_count')
ax.set_ylabel('vote_average')
ax.set_zlabel('popularity')

#Content dimension
fig = plt.figure(14)
ax = fig.add_subplot(111, projection='3d',title="Content variables")
ax.scatter(contentMovies[:,0],contentMovies[:,1],contentMovies[:,2],c=colorMap)
ax.set_xlabel('keywords')
ax.set_ylabel('genres')
ax.set_zlabel('runtime')

#Log content dimension
fig = plt.figure(15)
ax = fig.add_subplot(111, projection='3d',title="Content variables LOG")
ax.scatter(contentMoviesLog[:,0],contentMoviesLog[:,1],contentMoviesLog[:,2],c=colorMap)
ax.set_xlabel('keywords')
ax.set_ylabel('genres')
ax.set_zlabel('runtime')

#########################SCATTER MATRIX########################################
#scatter of the attributes aka scatter matrix
#Colors are green for non awarded and red for awarded

#Scatter of the matrix colored by variable isAwarded      
plt.figure(16)
pd.plotting.scatter_matrix(movieAttributes,figsize=(12,12),diagonal='kde',color=colorMap)

#Scatter of the log matrix colored by variable isAwarded  
plt.figure(17)
pd.plotting.scatter_matrix(movieLog,figsize=(12,12),diagonal='kde',color=colorMap)

#############################SAVE LOG DATA#####################################
#Finally a dataset with log transformation is created
moviesLogTransform = pd.concat([movieNames,movieLog,movieTarget],axis=1)
moviesLogTransform.to_csv('datasets/moviesCleanedImputatedLog.csv',sep='_',index=False)