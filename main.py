import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import linear_model

import matplotlib.pyplot as plt

#Preprocessing, cleaning
df = pd.read_csv("/home/ojus/Documents/binarytree/credit.csv",header=None)
df = df.replace('?', value= np.nan)
df = df.dropna(axis= 0, how= 'any')
cat = [0,3,4,5,6,8,9,11,12,15]

df = pd.get_dummies(df,columns= cat, drop_first= True)

X = df.drop(columns= ['15_-'])
Y = df[['15_-']]

#Generating Scores for different classifiers
logregScores = list()
svmScores = list()
treeScores = list()

for i in range(0,50) :
    
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.33)

    logreg = linear_model.LogisticRegression()
    logreg.fit(X_tr,Y_tr)
    logregScores.append(logreg.score(X_ts,Y_ts))

    supVecC = SVC(C= 10)
    supVecC.fit(X_tr,Y_tr)
    svmScores.append(supVecC.score(X_ts,Y_ts))
    
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_tr,Y_tr)
    treeScores.append(tree.score(X_ts,Y_ts))

#Plotting the Scores
fig,ax = plt.subplots()

ax.plot(logregScores,'k--',label='Logistic Regression')
ax.plot(svmScores,'k:',label='SVC')
ax.plot(treeScores,'k',label='Binary Decision Tree')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Training No.')
legend = ax.legend(loc='center right')
plt.show()