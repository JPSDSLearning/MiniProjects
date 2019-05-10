# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:55:47 2019

@author: Nagendra.B
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# Importing train and test dataset
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#display the head/first six records of the dataset using head()
train_data.head()
test_data.head()

#count the numberof observition missing data contains the dataset
train_data.isnull().sum()

#persons who survived and who don’t with a plot
sb.countplot('Survived',data=train_data)
plt.show()

#It is clear that the no of people survived is less than the number of people who died.

#Lets find the which category people more survived 
train_data.groupby(['Sex', 'Survived'])['Survived'].count()

'''It is clear that 233 female survived out of 344. And out of 577 male 109 survived.
 The survival ratio of female is much greater than that of male. 
 It can be seen clearly in following graph'''
 
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sb.countplot('Sex',hue='Survived',data=train_data,)
plt.show()

#check with other features

sb.countplot('Pclass', hue='Survived', data=train_data)
plt.title('Pclass: Sruvived vs Dead')
plt.show()

pd.crosstab([train_data.Sex,train_data.Survived],train_data.Pclass,margins=True)

print('Oldest person Survived was of:',train_data['Age'].max())
print('Youngest person Survived was of:',train_data['Age'].min())
print('Average person Survived was of:',train_data['Age'].mean())
print('Median person Survived was of:',train_data["Age"].median(skipna=True))
# median age is 28 (as compared to mean which is ~30)

train_data["Age"].fillna(28, inplace=True)

train_data['Initial']=0
for i in train_data:
    train_data['Initial']=train_data.Name.str.extract('([A-Za-z]+)\.') #extracting Name initials
    

pd.crosstab(train_data.Initial,train_data.Sex).T
#There are many names which are not relevant like Mr, Mrs etc. So I will replace them with some relevant names
train_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                               'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',
                                'Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    

pd.crosstab(train_data.Initial,train_data.Sex).T

train_data.groupby('Initial')['Age'].mean()

train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mr'),'Age']=33
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mrs'),'Age']=36
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Master'),'Age']=5
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Miss'),'Age']=22
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Other'),'Age']=46

train_data.Age.isnull().any()

f,ax=plt.subplots(1,2,figsize=(20,20))
train_data[train_data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived = 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_data[train_data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
ax[1].set_title('Survived = 1')
plt.show()

'''
First priority during Rescue is given to children and women, as the persons<5 are save by large numbers 
The oldest saved passanger is of 80 
The most deaths were between 20-40
'''
#SibSip feature indicates that whether a person is alone or with his family. Siblings=brother,sister, etc and Spouse= husband,wife

pd.crosstab([train_data.SibSp],train_data.Survived)

pd.crosstab(train_data.SibSp,train_data.Pclass)

'''
here are many interesting facts with this feature.
crosstabs shows that if a passanger is alone in ship with no siblings, survival rate is 34.5%.
The graph decreases as no of siblings increase.
This is interesting because, If I have a family onboard, I will save them instead of saving myself.
But there's something wrong, the survival rate for families with 5-8 members is 0%.
Is this because of PClass? Yes this is PClass, The crosstab shows that Person with SibSp>3 were all in Pclass3.
It is imminent that all the large families in Pclass3(>3) died.
'''

# proportion of "cabin" missing
round(687/len(train_data["PassengerId"]),4)
'''
77% of records are missing, which means that imputing information and using this variable for prediction is probably not wise.
We'll ignore this variable in our model.
'''

# proportion of "Embarked" missing
round(2/len(train_data["PassengerId"]),4)

'''
There are only 2 missing values for "Embarked", so we can just impute with the port where most people boarded.
'''

sb.countplot(x='Embarked',data=train_data,palette='Set2')
plt.show()

# Replace the missing values
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

'''
According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family.
For simplicity's sake (and to account for possible multicollinearity),
I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.
'''

## Create categorical variable for traveling alone

train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)

'''
I'll also create categorical variables for Passenger 
Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked").
'''
train_data = pd.get_dummies(train_data, columns=["Pclass"])
train_data = pd.get_dummies(train_data, columns=["Embarked"])
train_data=pd.get_dummies(train_data, columns=["Sex"])
train_data.drop('Sex_female', axis=1, inplace=True)

train_data.drop('PassengerId', axis=1, inplace=True)
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)
train_data.head(5)

train_data['IsMinor']=np.where(train_data['Age']<=16, 1, 0)


'''
Apply the same steps for test_data
'''



cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X_train=train_data[cols]
y_train=train_data['Survived']

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

logreg.score(X_train, y_train)


#Using 80-20 Split for Cross Validation

from sklearn.model_selection import train_test_split
train, test = train_test_split(train_data, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train=train[cols]
y_train=train['Survived']

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_train, y_train)

'''
The score for the new training sample (80% of original) is very close to the original performance, which is good!
Let's assess how well it scores on the 20% hold-out sample.
'''

logreg.fit(X_train, y_train)

X_test = test[cols]
y_test = test['Survived']

ytest_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.svm import SVC

svc = SVC(C = 0.1, gamma=0.1)
svc.fit(X_train, y_train)

result_train = svc.score(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

cols=["Age", "Fare", "TravelAlone", "Pclass_1",'Pclass_3', "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X_train=train_data[cols]
y_train=train_data['Survived']

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
random_forest.score(X_train, y_train)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

knn.score(X_train, y_train)
Y_pred = knn.predict(X_test)

knn.score(X_train, y_train)


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)

Y_pred = gnb.predict(X_test)

gnb.score(X_train, y_train)




