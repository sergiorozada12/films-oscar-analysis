import pandas as pd
import json

######################DEFINITION OF FUNCTIONS##################################

#This function will be used later, it takes a column of the dataframe where each
#value is a list of dictinaries, extracts the key values of the dictionaries and
#returns the list of words extracted
def keyValueExtraction(column):
    listFinal=[]
    for listDictionaries in column:
        listInRow=[]
        for item in listDictionaries:
            listInRow.append(item['name'])
        listFinal.append(listInRow)
    return listFinal

##################################LOAD DATA####################################

#We load the dataframes from movies and oscar's winners
movies=pd.read_csv('datasets/sources/tmdb_5000_movies.csv')
winners=pd.read_csv('datasets/sources/database.csv')

################################FIRST CHANGE##################################
#We are going to transform the columns which are json into a list containing the
#values of the Json. Eg: extract keywords from json format to a list

#First of all we use json library to transform json strings into list of dictionaries
movies.keywords=movies.keywords.apply(lambda x: json.loads(x))
movies.production_companies=movies.production_companies.apply(lambda x: json.loads(x))
movies.genres=movies.genres.apply(lambda x: json.loads(x))

#We asign the list of keywords, producers and languages to the proper dataframe column
movies.keywords=keyValueExtraction(movies.keywords)
movies.production_companies=keyValueExtraction(movies.production_companies)
movies.genres=keyValueExtraction(movies.genres)

#####################################SECOND CHANGE#############################
#We are remoiving the following columns
#Overview, Original_title, tagline, spoken_languages, id, homepage and status for
#being redundant information or not applicable to the analyisis which is going
#to be developed
movies.pop('overview')
movies.pop('original_title')
movies.pop('tagline')
movies.pop('spoken_languages')
movies.pop('id')
movies.pop('homepage')
movies.pop('status')
movies.pop('production_countries')
movies.pop('original_language')

##################################THIRD CHANGE################################
#We are trying to add a colum with a YES or NO wheter the film has been awarded
#or not with a Oscar.
awarded=[]

#We use the strip function to trim spaces in both sides and make Strings comparable later on
awardedFilms=winners.Name.apply(lambda x: x.strip())

#For each film we check wether it is in the winners list or not for films and also
#for individuals awards and we carry out a boolean register of winner or not.
for film in movies.title:
    hasWon=False
    film=film.strip()
    for winner in awardedFilms:
        if (film==winner):
            hasWon=True
            
    if(hasWon):
        awarded.append(True)
    else:
        awarded.append(False)

#Once we have the list of winners we append it to a new column of our dataset
movies['isAwarded']=awarded


################################FOURTH CHANGE##################################
#We parse the release date which is a String into a pandas time object
movies.release_date=movies.release_date.apply(lambda x: pd.to_datetime(x))

###############################FIFTH CHANGE####################################
#To make economic data from different periods of time comparable, we have to 
#correct the inflation effect.

#Dataset with CPI values since 1910
cpi=pd.read_csv('datasets/sources/cpi.csv',sep=';')
cpi=cpi.set_index('year')
currentCpi=cpi.cpi[2017]

#Correct the inflation effect on budget and revenues columns
inflation=pd.DataFrame()
inflation['year']=movies.release_date.apply(lambda x: x.year)
inflation['inflation']=inflation.year.copy()
inflation['inflation']=inflation.inflation.fillna(2017)
inflation['inflation']=inflation.inflation.apply(lambda x: (currentCpi-cpi.cpi[x])/currentCpi )

#Apply the correction in budget and revenues
budget=[]
revenue=[]
for item in range(movies.shape[0]):
    if(inflation.inflation[item]!=0):
        budget.append(movies.budget[item]/inflation.inflation[item])
        revenue.append(movies.revenue[item]/inflation.inflation[item])
    else:
        budget.append(movies.budget[item])
        revenue.append(movies.revenue[item])

movies.budget=budget
movies.revenue=revenue

################################SIXTH CHANGE###################################
#Add a profit column so that we can analyze how much films are earning
movies['profit']=movies.revenue-movies.budget

################################SAVE CHANGES###################################

#Once we performed all the changes we save the dataset. Notice that no value 
#imputation or modification has been made, this script only gathers all interesting
#data for the application
movies.to_csv('datasets/moviesPrepared.csv',sep='_',index=False)
