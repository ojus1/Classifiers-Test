import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import linear_model

import matplotlib.pyplot as plt

#Preprocessing, cleaning
df = pd.read_csv("/home/ojus/Documents/classifiers_plot/credit.csv",header=None)
df = df.drop([5])
df = df.replace('?', value= np.nan)
df = df.dropna(axis= 0, how= 'any')
cat = [0,3,4,5,6,8,9,11,12,15]

df = pd.get_dummies(df,columns= cat, drop_first= True)
df.to_csv('cleaned_credit.csv')

X = df.drop(columns= ['15_-'])
Y = df[['15_-']]

#Generating Scores for different classifiers
logregScores = list()
forestScores = list()
treeScores = list()

for i in range(0,50) :
    
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.25)

    
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_tr,Y_tr)
    logregScores.append(cross_val_score(logreg,X_ts,Y_ts).mean())

    forest = AdaBoostClassifier(n_estimators = 100)
    forestScores.append(cross_val_score(forest,X_ts,Y_ts).mean())
    
    
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_tr,Y_tr)
    treeScores.append(cross_val_score(tree,X_ts,Y_ts).mean())

#Plotting the Scores
fig,ax = plt.subplots()

ax.plot(logregScores,'k--',label='Logistic Regression')
ax.plot(forestScores,'k:',label='Adaboost forest')
ax.plot(treeScores,'k',label='Binary Decision Tree')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Training No.')
legend = ax.legend(loc='center right')
fig.show()
