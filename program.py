import ast
import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict

#TODO: check for better smothing function, not 0.01
def tokenize_description(desc):
    if pd.isna(desc):
        return set()
    return set( re.findall(r'\b\w+\b', desc.upper())  )
start = time.time()

data=pd.read_csv('games_esential_only.csv')
data['genres']=data['genres'].apply(ast.literal_eval)
print("data was read")

wantedCheking=['Action','Free To Play','Adventure','Massively Multiplayer', 'Strategy', 'Indie', 'RPG', 'Casual', 'Simulation',  'Sports', 'Racing']

#'Action','Free To Play','Adventure','Massively Multiplayer', 'Strategy', 'Indie', 'RPG', 'Casual', 'Simulation',  'Sports', 'Racing'
splitPoint=input(f"Sunt {len(data)} de lini in setul de date, cate linii vrei sa fie date de training, nu introduce nimic daca vrei ca setul de training si cel de testing sa fie acelasi")
if(splitPoint!=""):
    splitPoint=int(splitPoint)
    allData=False
else:
    allData=True

if(allData):
    TrainingData=data
    TestingData=data
else:
    TrainingData=data.iloc[:splitPoint]
    TestingData=data.iloc[splitPoint:]

#------------
#PRocessData
#------------
numOfGamesPerType={genre:0 for genre in wantedCheking}
tokens={}

for row in TrainingData.itertuples():

    currentGameGenres=set()
    for j in row.genres :
        if j in wantedCheking:
            currentGameGenres.add(j)
            numOfGamesPerType[j]+=1

    currentGameWords=tokenize_description(row.detailed_description)


    for word in currentGameWords:
        if word in tokens:
            tokens[word]['total']+=1
        else:
            tokens[word]={'total':1}
        
        for gameGenres in currentGameGenres:
            if gameGenres in tokens[word]:
                tokens[word][gameGenres]+=1
            else:
                tokens[word][gameGenres]=1

tokens = {k: v for k, v in tokens.items() if v['total'] > 10 and k not in {"a","an","and","or","the","of","in","on","to","for","by","at","is","are","be","was","were"}}

x=0
ans=0
print("data was procesed")
#------------
#CHECK DATA
#------------
priors={}
correct=0
incorrect=0

TruePositive=defaultdict(lambda: 0)
TrueNegative=defaultdict(lambda: 0)
FalsePositive=defaultdict(lambda: 0)
FalseNegative=defaultdict(lambda: 0)
for j in wantedCheking :
    priors[j]=np.log(numOfGamesPerType[j]/len(TrainingData))

def testWord(str):
    probPerType=priors.copy()
    
    gameDescriptionTokens=tokenize_description(str)
    for word in gameDescriptionTokens: 
        if word in tokens:
            p_word = tokens[word]['total'] / len(TrainingData)   
            for genre in probPerType: 
                count = tokens[word].get(genre, 0.01)         
                p_word_given_genre = count / (numOfGamesPerType[genre] +0.01)
                probPerType[genre] +=np.log(p_word_given_genre / p_word)
    
  
  
    PredictedValues={type for type,logProb in probPerType.items() if logProb>=0}
    return PredictedValues
print("Introduce o descriere ca sa o testezi, apasa enter pentru a testa pe setul de date ramase")
str="cevaCeNuENull"
while(str!=""):
    str=input()
    PrValues=testWord(str)
    print(PrValues)

for row in TestingData.itertuples():

    PredictedValues=testWord(row.detailed_description)
    
    RealValues={j for j in row.genres if j in wantedCheking}


    for type in wantedCheking:
        if type in RealValues and type in PredictedValues:
            TruePositive[type] += 1
            correct+=1
        elif type not in RealValues and type not in PredictedValues:
            TrueNegative[type] += 1
            correct+=1
        elif type in RealValues and type not in PredictedValues:
            FalseNegative[type] += 1 
            incorrect+=1
        elif type not in RealValues and type in PredictedValues:
            FalsePositive[type] += 1   # CORECT
            incorrect+=1
    if RealValues==PredictedValues:
        ans+=1
    if x%100 == 0:
        print(f"\nGame {x}: {row.name}")
        print(f"Current Correct Values:{ans}")
        print(f"Expected  :{RealValues}")
        print(f"Predicted :{PredictedValues}")
    x+=1
print(f"CorrectAnswers:{ans}")
print(f"Correctness{correct/(correct+incorrect)*100:.2f}")
print(f"Correctess per type:")
for i in wantedCheking:
    print(f"{i}: "
      f" Sensitivity={TruePositive[i]/(TruePositive[i]+FalseNegative[i])*100:.2f}"
      f" Specificity={TrueNegative[i]/(TrueNegative[i]+FalsePositive[i])*100:.2f}"
      f" Precision={TruePositive[i]/(TruePositive[i]+FalsePositive[i])*100:.2f}"
      f" Accuracy={(TruePositive[i]+TrueNegative[i])/(TruePositive[i]+TrueNegative[i]+FalseNegative[i]+FalsePositive[i])*100:.2f}")
print(f"ExpectedValueIfRandom:{int(x/(2**len(wantedCheking)))}")
#random=43



end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")