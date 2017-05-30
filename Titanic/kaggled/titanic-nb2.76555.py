import pandas as pd
import numpy as np

filename="train.csv"

def new_train_csv():
	#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	#Sliced dataset of -100 random items from original training data
	dataframe1  = pd.read_csv("new-train.csv",usecols=(1,2,4,5,6,7,9))
	dataframe1.replace("female",0,inplace=True)
	dataframe1.replace("male",1,inplace=True)
	dataframe2=dataframe1.fillna(1000)
	features_training=dataframe2.iloc[:,[1,2,3,4,5]]
	labels_training=dataframe1.iloc[:,0]
	return (features_training,labels_training)
def new_test():
	#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	#Sliced dataset of 100 random items from original training data
	test_data  = pd.read_csv("h100.csv",usecols=(1,4,5,6,9))
	test_data.replace("female",0,inplace=True)
	test_data.replace("male",1,inplace=True)
	test_data2 = test_data.fillna(1000)
	test_data3 = test_data2.iloc[:,[0,1,2,3,4]]
	labels_test = test_data2.iloc[:,0]
	return(test_data3,labels_test)

def train_csv():
	#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	dataframe1  = pd.read_csv("train.csv",usecols=(1,2,4,5,6,7,9))
	#print("dataframe1 training:\n",dataframe1.head(10))
	dataframe1.replace("female",0,inplace=True)
	dataframe1.replace("male",1,inplace=True)
	dataframe2=dataframe1.fillna(1000)
	features_training=dataframe2.iloc[:,[1,2,3,4,5,6]]
	labels_training=dataframe1.iloc[:,0]
	#print("Features training:\n",features_training.head(10))
	#print("Labels training:\n",labels_training.head(10))
	return (features_training,labels_training)


def test():
	#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	test_data  = pd.read_csv("test.csv",usecols=(1,3,4,5,6,8))
	test_data.replace("female",0,inplace=True)
	test_data.replace("male",1,inplace=True)
	test_data2 = test_data.fillna(1000)
	test_data3 = test_data2.iloc[:,[0,1,2,3,4,5]]
	labels_test = test_data2.iloc[:,0]
	#print("test_data3:\n",test_data3.head(5))
	return (test_data3,labels_test)
	

(features_training,labels_training) = train_csv()
(features_test,labels_test) = test()

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(features_training,labels_training)

results = clf.predict(features_test)
from sklearn.metrics import accuracy_score

count=892
print("PassengerId,Survived")
for i in results:
	print(count,",",i)
	count += 1

#print("Accuracy Score only you are using new csv and new test variables:",accuracy_score(results,labels_test))
