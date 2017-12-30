import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#################################LOAD THE DATA#################################
#Load the data
movies = pd.read_csv('datasets/moviesCleanedImputatedLog.csv',sep='_',encoding = "ISO-8859-1")
print(movies.head())

#################################PREPARE DATA##################################
#Preparation of the data for the analysis
movieNames = movies.title
movieTarget = movies.isAwarded
movieAttributes = movies[['budget','revenue','profit','genres','keywords','production_companies','runtime','vote_count','vote_average','popularity']]
features = list(movieAttributes)

#Data should be normalized
movieAttributes = (movieAttributes-movieAttributes.mean())/movieAttributes.std()

#Data as array
movieAttributesArr = movieAttributes.as_matrix()
movieTargetArr = np.asarray(movieTarget)

###########################KMEANS TOTAL########################################
#Clustering of all features
movieClustering = KMeans(init='random',n_clusters = 2,n_init=10).fit_predict(movieAttributesArr)

#PCA dimensionality reduction for visualization
pca = PCA(n_components = 2)
pca.fit(movieAttributesArr)
projectedAttributesPca = pca.transform(movieAttributesArr)

#Data is colored given isAwarded and cluster number
colorMap = []
colorMapKmeans = []

for i in range(len(movieTargetArr)):
    
    if (movieTargetArr[i] == 0):
        colorMap.append('r')
    else:
        colorMap.append('g')
        
    if (movieClustering[i] == 0):
        colorMapKmeans.append('r')
    else:
        colorMapKmeans.append('g')
        
#Scatter plot of the results with true labelling and kmeans
plt.figure(1)
plt.title("PCA with true class labeling")
plt.scatter(projectedAttributesPca[:,0],projectedAttributesPca[:,1],c=colorMap)

plt.figure(2)
plt.title("PCA with kmeans labeling")
plt.scatter(projectedAttributesPca[:,0],projectedAttributesPca[:,1],c=colorMapKmeans)

############################KMEANS ECONOMIC####################################
#Clustering of economic features and 3d plot
economicFeatures = movieAttributes[['revenue','profit','budget']].as_matrix()
economicClustering = KMeans(init='random',n_clusters = 2,n_init=10).fit_predict(economicFeatures)

colorMapKmeansEc = []

for i in range(len(movieTargetArr)):       
    if (economicClustering[i] == 0):
        colorMapKmeansEc.append('r')
    else:
        colorMapKmeansEc.append('g')

#scatter in 3d
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(economicFeatures[:,0],economicFeatures[:,1],economicFeatures[:,2],c=colorMap)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(economicFeatures[:,0],economicFeatures[:,1],economicFeatures[:,2],c=colorMapKmeansEc)

#################################KMEANS SOCIAL#################################
#Clustering of social features and 3d plot
socialFeatures = movieAttributes[['vote_count','vote_average','popularity']].as_matrix()
socialClustering = KMeans(init='random',n_clusters = 2,n_init=10).fit_predict(socialFeatures)

colorMapKmeansSoc = []

for i in range(len(movieTargetArr)):       
    if (socialClustering[i] == 0):
        colorMapKmeansSoc.append('r')
    else:
        colorMapKmeansSoc.append('g')

#scatter in 3d
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(socialFeatures[:,0],socialFeatures[:,1],socialFeatures[:,2],c=colorMap)

fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(socialFeatures[:,0],socialFeatures[:,1],socialFeatures[:,2],c=colorMapKmeansSoc)

##############################KMEANS# CONTENT##########################################
#Clustering of content based features and 3d plot
contentFeatures = movieAttributes[['keywords','genres','runtime']].as_matrix()
contentClustering = KMeans(init='random',n_clusters = 2,n_init=10).fit_predict(contentFeatures)

colorMapKmeansCon = []

for i in range(len(movieTargetArr)):       
    if (contentClustering[i] == 0):
        colorMapKmeansCon.append('r')
    else:
        colorMapKmeansCon.append('g')

#scatter in 3d
fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(contentFeatures[:,0],contentFeatures[:,1],contentFeatures[:,2],c=colorMap)

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(contentFeatures[:,0],contentFeatures[:,1],contentFeatures[:,2],c=colorMapKmeansCon)