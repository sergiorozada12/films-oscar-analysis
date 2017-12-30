import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

##############################LOAD DATA########################################
#Load the data
movies = pd.read_csv('datasets/moviesCleanedImputatedLog.csv',sep='_',encoding = "ISO-8859-1")
print(movies.head())

##############################PREPARE DATA#####################################
#Data is prepared for the analysis
movAtt = movies[['budget','revenue','profit','genres','keywords','production_companies','runtime','vote_count','vote_average','popularity']]
columnNames = list(movAtt)
movTarget = movies.isAwarded

#Data should be normalized
movAtt = (movAtt-movAtt.mean())/movAtt.std()
movAttMat = movAtt.as_matrix()

##################################PCA##########################################

#PCA dimensionality reduction is performed to better understand the data
pca = PCA(n_components = 10)
pca.fit(movAttMat)
projectedAttributesPca = pca.transform(movAttMat)

#Data is colored given isAwarded target
colorMap = []
for target in movTarget:
    if target==0:
        colorMap.append('r')
    else:
        colorMap.append('g')
        
#Plot the pca projections
fig = plt.figure(1,figsize=(10,10))
ax = fig.add_subplot(111, projection='3d',title="Principal Components Analysis projection")
ax.scatter(projectedAttributesPca[:,0],projectedAttributesPca[:,1],projectedAttributesPca[:,2],color=colorMap)
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.set_zlabel('Third component')   

plt.figure(2,figsize=(10,10))
plt.scatter(projectedAttributesPca[:,0],projectedAttributesPca[:,1],color=colorMap)
plt.title('Principal Components Analysis projection')
plt.xlabel('First component')
plt.ylabel('Second component')


#Weights of first component
comp1Weights = np.asarray(pca.components_[0])[np.argsort(np.abs(pca.components_[0]))[::-1]]
comp1Names = np.asarray(columnNames)[np.argsort(np.abs(pca.components_[0]))[::-1]]

print("\nFIRST COMPONENT:")
for i in range(len(comp1Weights)):
    print(" - Variable ",comp1Names[i]," has a weight: ",comp1Weights[i])

#Weights of second component
comp2Weights = np.asarray(pca.components_[1])[np.argsort(np.abs(pca.components_[1]))[::-1]]
comp2Names = np.asarray(columnNames)[np.argsort(np.abs(pca.components_[1]))[::-1]]

print("\nSECOND COMPONENT:")
for j in range(len(comp2Weights)):
    print(" - Variable ",comp2Names[j]," has a weight: ",comp2Weights[j])
    
#Explained variance
print("\nEXPLAINED VARIANCE:")
for i in range(10):
    print("Component: ",i+1,"explains: ", pca.explained_variance_[i]*10,"% of the variance")
print("\n")
    
################################MDS############################################
#Let's check if a MDS helps us distinguishing classes by distances between them
#Perhaps we find some clusters
#Randomly, sample 400 movies of the dataset
idxMds = np.random.random_integers(0,len(movTarget)-1,1000)
movTargetMds = np.asarray(movTarget)[idxMds]

#Distance matrix is calculated for all parameters (normalized)
distMatrix = euclidean_distances(movAttMat[idxMds,:],movAttMat[idxMds,:])

#MDS model is applied
mds = manifold.MDS(n_components = 2, max_iter=3000, n_init=1, dissimilarity="precomputed")
projectedAttributesMds = mds.fit_transform(distMatrix)

#Color of the data given isAwarded
colorMapMds = []
for target in movTargetMds:
    if target==0:
        colorMapMds.append('r')
    else:
        colorMapMds.append('g')

#scatter plot the results
plt.figure(3,figsize=(10,10))
plt.title("Multi Dimensional Scaling projection (1000 movies)")
plt.scatter(projectedAttributesMds[:,0],projectedAttributesMds[:,1],c=colorMapMds)

#############################LDA###############################################

#LDA is also applied and we obtain the best separation (True/False) for movie attributes
lda = LDA(n_components=1)
projectedAttributesLda = lda.fit_transform(movAttMat,np.asarray(movTarget))
coeff = lda.coef_[0]

#Weights of second component
ldaWeights = np.asarray(coeff)[np.argsort(np.abs(coeff))[::-1]]
ldaNames = np.asarray(columnNames)[np.argsort(np.abs(coeff))[::-1]]

print("\nLDA COMPONENTS:")
for j in range(len(ldaWeights)):
    print(" - Variable ",ldaNames[j]," has a weight: ",ldaWeights[j])