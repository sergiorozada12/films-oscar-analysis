import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval as le

#################################FUNCTION######################################
#Counts the number of words and sorts the output
def countFromList(listToConvert):
    words = []
    values = []
    
    #For each list of words 
    for string in listToConvert:
        #Cast from string to list
        listFromString = le(string)
        #For each word in the list
        for reference in listFromString:
            #If already exists: increment count value
            if reference in words:
                i = words.index(reference)
                values[i] += 1
            #Else: create the count value
            else:
                words.append(reference)
                values.append(1)
    
    #As a matrix with the output sorted
    mat = np.transpose(np.array([words,values]))
    return mat[np.argsort(mat[:,1].astype(int))[::-1]]

###############################################################################
#Calculate the weight of each word in the list
def calculateWeightWords(total,awardedMat,notAwardedMat):
    #Now map the weight of the words into 1-100 count
    awarded = awardedMat[:,1].astype(int)
    adjustAw = 100/awarded[0]
    keyVarAw = list(map(lambda x: x*adjustAw,awarded))
    
    notAwarded = notAwardedMat[:,1].astype(int)
    adjustNaw = 100/notAwarded[0]
    keyVarNaw = list(map(lambda x: x*adjustNaw,notAwarded))
    
    key = []
    weights = []
    
    #For each word in the total list
    for word in total[:,0]:
        #If word is in both lists, weight depends in relation between word counts in awarded and non-awarded films
        if ((word in awardedMat[:,0])&(word in notAwardedMat[:,0])):
            key.append(word)
            ratio = keyVarAw[np.where(awardedMat[:,0]==word)[0][0]]/keyVarNaw[np.where(notAwardedMat[:,0]==word)[0][0]]
            
            #Ratio shows keyword is largely more named in awarded films
            if ratio>1.4:
                weights.append(3*keyVarAw[np.where(awardedMat[:,0]==word)[0][0]])
            #Ratio shows keyword is largely more named in non-awarded films
            elif ratio<0.8:
                weights.append(keyVarAw[np.where(awardedMat[:,0]==word)[0][0]])
            #Ratio shows keyword is named in awarded and non-awarded films equally
            else:
                weights.append(2*keyVarAw[np.where(awardedMat[:,0]==word)[0][0]])
        #If word only appears in awarded films, it weights a lot   
        elif (word in awardedMat[:,0]):
            key.append(word)
            weights.append(4*keyVarAw[np.where(awardedMat[:,0]==word)[0][0]])
        #If word only appears in non-awarded films, it weights nothing
        else:
            key.append(word)
            weights.append(1)
            
    return np.transpose(np.array([key,weights]))

###############################################################################
#Calulate the weight of each total list of words
def calculateNewColumn(weights, oldColumn):
    newColumn = []
    keys = weights[:,0]
    values = weights[:,1].astype(float)
    
    #For each list of words the score is calculated ( sum of all words)
    for element in oldColumn:
        element = le(element)
        score = 0
        #For each word we add the weight of the word
        for item in element:
            score += values[np.where(keys==item)[0][0]]
        newColumn.append(score)
    
    return newColumn
            
#############################LOAD DATA#########################################
#Load of the prepared dataset
movies=pd.read_csv('datasets/moviesPrepared.csv',sep='_',encoding = "ISO-8859-1")
print(movies.head())

##############################FE KEYWORDS######################################
#Creation of counts for words in all, awarded and non-awarded
keywords = countFromList(movies.keywords)
awardedKeywords = countFromList(movies.keywords[movies.isAwarded==True])
notAwardedKeywords = countFromList(movies.keywords[movies.isAwarded==False])

#Plot the results in an horizontal bar chart for the 15 most mentioned words
N = 15
plt.figure(1)
plt.title('Most quoted keywords')
toPlotKeywords = keywords[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotKeywords)
plt.yticks(range(N),keywords[0:N,0])

plt.figure(2)
plt.title('Most quoted keywords in awarded films')
toPlotKeywordsAwarded = awardedKeywords[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotKeywordsAwarded)
plt.yticks(range(N),awardedKeywords[0:N,0])

plt.figure(3)
plt.title('Most quoted keywords in non-awarded films')
toPlotKeywordsNotAwarded = notAwardedKeywords[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotKeywordsNotAwarded)
plt.yticks(range(N),notAwardedKeywords[0:N,0])

#Weights for the words and value of each list of words
keywordWeights = calculateWeightWords(keywords,awardedKeywords,notAwardedKeywords)
movies.keywords = calculateNewColumn(keywordWeights,movies.keywords)

##############################FE PRODUCERS#####################################
#Creation of counts for words in all, awarded and non-awarded
producers = countFromList(movies.production_companies)
awardedProducers = countFromList(movies.production_companies[movies.isAwarded==True])
notAwardedProducers = countFromList(movies.production_companies[movies.isAwarded==False])

#Plot the results in an horizontal bar chart for the 15 most mentioned words
plt.figure(4)
plt.title('Most quoted producers')
toPlotProducers = producers[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotProducers)
plt.yticks(range(N),producers[0:N,0])

plt.figure(5)
plt.title('Most quoted producers in awarded films')
toPlotProducersAwarded = awardedProducers[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotProducersAwarded)
plt.yticks(range(N),awardedProducers[0:N,0])

plt.figure(6)
plt.title('Most quoted producers in non-awarded films')
toPlotProducersNotAwarded = notAwardedProducers[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotProducersNotAwarded)
plt.yticks(range(N),notAwardedProducers[0:N,0])

#Weights for the words and value of each list of words
producerWeights = calculateWeightWords(producers,awardedProducers,notAwardedProducers)
movies.production_companies = calculateNewColumn(producerWeights,movies.production_companies)

##############################FE GENRES########################################
#Creation of counts for words in all, awarded and non-awarded
genres = countFromList(movies.genres)
awardedGenres = countFromList(movies.genres[movies.isAwarded==True])
notAwardedGenres = countFromList(movies.genres[movies.isAwarded==False])

#Plot the results in an horizontal bar chart for the 15 most mentioned words
plt.figure(7)
plt.title('Most quoted genres')
toPlotGenres = genres[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotGenres)
plt.yticks(range(N),genres[0:N,0])

plt.figure(8)
plt.title('Most quoted genres in awarded films')
toPlotGenresAwarded = awardedGenres[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotGenresAwarded)
plt.yticks(range(N),awardedGenres[0:N,0])

plt.figure(9)
plt.title('Most quoted genres in non-awarded films')
toPlotGenresNotAwarded = notAwardedGenres[0:N,1].astype(int)
plt.gca().invert_yaxis()
plt.barh(range(N),toPlotGenresNotAwarded)
plt.yticks(range(N),notAwardedGenres[0:N,0])

#Weights for the words and value of each list of words
genresWeights = calculateWeightWords(genres,awardedGenres,notAwardedGenres)
movies.genres = calculateNewColumn(genresWeights,movies.genres)

##################################FINAL SCREENING##############################
#Plot of some general statistics
print ( "\nAwarded movies:")
print (movies[['keywords','production_companies','genres']][movies.isAwarded==True].describe())

print ( "\nNon-awarded movies:")
print (movies[['keywords','production_companies','genres']][movies.isAwarded==False].describe())

#################################SAVE DATASET##################################
#Finally we save the dataset
movies.to_csv('datasets/moviesEngineered.csv',sep='_',index=False)