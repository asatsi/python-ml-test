import pandas as pd
import numpy as np

filename="train.csv"

#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

dataframe1  = pd.read_csv("train.csv",usecols=(1,2,4,5,6,7,9))

dataframe1.replace("female",1,inplace=True)
dataframe1.replace("male",1,inplace=True)
dataframe1.dropna(inplace=True)


training=dataframe1.iloc[:,[1,2,3,4,5,6]]
labels=dataframe1.iloc[:,0]


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(dataframe1,labels)
