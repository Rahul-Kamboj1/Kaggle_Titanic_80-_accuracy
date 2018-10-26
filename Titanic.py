import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

def bar_char(feature):
    survived=df_train[df_train['Survived']==1][feature].value_counts()
    dead=df_train[df_train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])

    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
	
#Check for plots to analyze the relations b/w fetature and output 	
bar_char('Sex')
bar_char('Pclass')
bar_char('SibSp')
bar_char('Parch')
bar_char('Embarked')


train_test_data=[df_train,df_test]

#extract the titleslie Mr, Mrs,Ms etc
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract(r'([a-zA-Z]+)\.')

title_mapping={'Mr':0,'Miss':1,'Mrs':2,'Master':0,'Dr':3,'Rev':3,'Major':0,'Col':0,'Mlle':3,'Sir':0,'Don':0,'Ms':1,'Countess':3,'Capt':0,'Mme':3,'Lady':2,'Jonkheer':3}
sex_mapping={'male':0,'female':1}

for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Sex']=dataset['Sex'].map(sex_mapping)
	
bar_char('Title')

#Remove the name column
df_train.drop(labels='Name',axis=1,inplace=True)
df_test.drop(labels='Name',axis=1,inplace=True)

#Filling the null values of age with mean  group by Title
df_train['Age'].fillna(df_train.groupby('Title')['Age'].transform("mean"), inplace=True)
df_test['Age'].fillna(df_test.groupby('Title')['Age'].transform("mean"), inplace=True)


#Analyzing surival chance vs age Kde plot.

facet=sns.FacetGrid(df_train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,df_train['Age'].max()))
facet.add_legend()
plt.show()


#Having closer look at the plot in age range (0,20)
facet=sns.FacetGrid(df_train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,df_train['Age'].max()))
facet.add_legend()
plt.xlim(0,20)

#Having closer look at the plot in age range (20,40)
facet=sns.FacetGrid(df_train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,df_train['Age'].max()))
facet.add_legend()
plt.xlim(20,40)

#Having closer look at the plot in age range (40,60)
facet=sns.FacetGrid(df_train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,df_train['Age'].max()))
facet.add_legend()
plt.xlim(40,60)


#Having closer look at the plot in age range (60,80)
facet=sns.FacetGrid(df_train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,df_train['Age'].max()))
facet.add_legend()
plt.xlim(60,80)


#Binning (conerting numerical data into Categorical values), based on analaysis given above.
for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26),'Age']=1
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36),'Age']=2
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62),'Age']=3
    dataset.loc[dataset['Age']>62,'Age'] =4

#from this plot we see that most of passengers embarked from Sex
sns.countplot(x='Embarked',data=df_train)

#So fill null values with S and assign number to it.
embar_mapping={'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    dataset['Embarked']=dataset['Embarked'].map(embar_mapping)
df_test['Title'].fillna(2,inplace=True)


#Filling missing fare by mean of fare of same group
df_train['Fare'].fillna(df_train.groupby('Pclass')['Fare'].transform('mean'),inplace=True)
df_test['Fare'].fillna(df_test.groupby('Pclass')['Fare'].transform('mean'),inplace=True)

#Clssifying the tickets into categories
for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17,'Fare']=0
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30),'Fare']=1
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare']=2
    dataset.loc[dataset['Fare']>100, 'Fare']=3
	
#fetching first column  of the Cabin	
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].str[:1]

cabin_mapping={'A':0.0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)

df_train['Cabin'].fillna(df_train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
df_test['Cabin'].fillna(df_test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)


df_train['FamilySize']=df_train["SibSp"]+df_train['Parch']+1
df_test['FamilySize']=df_test["SibSp"]+df_test['Parch']+1

df_train.drop('Ticket',axis=1,inplace=True)
df_test.drop('Ticket',axis=1,inplace=True)


train_data=df_train.drop(['Survived','PassengerId'],axis=1)

target=df_train['Survived']

from sklearn.svm import SVC

from sklearn.model_selection import  KFold
from sklearn.model_selection import cross_val_score

kfold=KFold(n_splits=10,shuffle=True,random_state=0)

clf=SVC()
scoring='accuracy'
score=cross_val_score(clf,train_data,target,cv=kfold,n_jobs=1,scoring=scoring)
print(score)
print(np.mean(score)*100,2)


clf=SVC()
clf.fit(train_data,target)

test_data=df_test.drop(['PassengerId'],axis=1)
prediction=clf.predict(test_data)

submission=pd.DataFrame({"PassengerId":df_test['PassengerId'],
                        "Survived":prediction})

submission.to_csv('submission.csv',index=False)						

	
