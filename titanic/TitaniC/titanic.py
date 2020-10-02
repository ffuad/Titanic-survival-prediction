# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:45:52 2020

@author: Fuad
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Importing Classifier Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    


train = pd.read_csv('D:/titanic/datasets/train.csv')
test = pd.read_csv('D:/titanic/datasets/test.csv')


"""print(train.head(5))
train.shape
test.shape
print(train.info())
print(test.info())
print(train.isnull().sum())
print(test.isnull().sum())
"""

sns.set()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    

'''bar_chart('Sex')
bar_chart('Sex')
'''

train_testdata = [train,test]

for dataset in train_testdata:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)

#train['Title'].value_counts()


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_testdata:
    dataset['Title'] = dataset['Title'].map(title_mapping)
'''
train.head()
test.head()
'''
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)

sex_mapping = {"male": 0, "female": 1}
for dataset in train_testdata:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)



# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.groupby("Title")["Age"].transform("median")
#train.head()
'''
facet = sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
'''
#plt.show()

for dataset in train_testdata:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

'''Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
'''
for dataset in train_testdata:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    
'''facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.show()
'''

for dataset in train_testdata:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

#train.Cabin.value_counts()
for dataset in train_testdata:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

'''
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
'''



embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_testdata:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#train.head(5)


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_testdata:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)  
    
#train.info()
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

'''
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
'''
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_testdata:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
    
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)


train_data = train.drop('Survived', axis=1)
target = train['Survived']
final_test = test.drop(['PassengerId'], axis=1)

#train_data.shape, target.shape    


#LOGISTIC_REGRESSION
logreg = LogisticRegression()
logreg.fit(train_data, target)
pred_target = logreg.predict(final_test)
acc_log = round(logreg.score(train_data, target)*100, 2)
#print(acc_log)

#support vector machines
svc = SVC()
svc.fit(train_data, target)
pred_target = svc.predict(final_test)
acc_svc = round(svc.score(train_data, target)*100,2)
#print(acc_svc)

#k neares neigbour
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_data, target)
pred_target = knn.predict(final_test)
acc_knn = round(knn.score(train_data, target)*100,2)
acc_knn
    
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, target)
pred_target = decision_tree.predict(final_test)
acc_decision_tree = round(decision_tree.score(train_data, target)*100,2)
acc_decision_tree    

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_data, target)
acc_random_forest = round(random_forest.score(train_data, target)*100,2)
acc_random_forest
pred_target = random_forest.predict(final_test)


models = pd.DataFrame({    
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree]})

#models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
        "Survived": pred_target
})

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
submission.head()



    
