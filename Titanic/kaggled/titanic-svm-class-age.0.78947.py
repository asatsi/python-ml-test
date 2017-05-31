import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

filename="train.csv"

def generate_dataframe(dataframe1,csvtype):
	print("dataframe1 training:\n",dataframe1.head(10))

	#set Sex as female = 0 and male = 1
	dataframe1.replace("female",0,inplace=True)
	dataframe1.replace("male",1,inplace=True)
	
	#Find Masters and assign mean value of Masters age to the missing ones
	master_mean = dataframe1[dataframe1.Name.str.contains("Master.")].Age.mean()
	mister_mean = dataframe1[dataframe1.Name.str.contains("Mr.")].Age.mean()
	mrs_mean = dataframe1[dataframe1.Name.str.contains("Mrs.")].Age.mean()
	miss_mean = dataframe1[dataframe1.Name.str.contains("Miss.")].Age.mean()
	print("master_mean value is+++++++++++",master_mean)
	print("mister_mean value is+++++++++++",mister_mean)
	print("mrs_mean value is+++++++++++",mrs_mean)
	print("miss_mean value is+++++++++++",miss_mean)

	dataframe1.loc[(dataframe1.Sex == 1) & dataframe1.Name.str.contains("Mr\.") & dataframe1.Age.isnull(),"Age"] = mister_mean
	dataframe1.loc[(dataframe1.Sex == 0) & dataframe1.Name.str.contains("Mrs\.") & dataframe1.Age.isnull(),"Age"] = mrs_mean
	dataframe1.loc[(dataframe1.Sex == 1) & dataframe1.Name.str.contains("Master\.") & dataframe1.Age.isnull(),"Age"] = master_mean
	dataframe1.loc[(dataframe1.Sex == 0) & dataframe1.Name.str.contains("Miss\.") & dataframe1.Age.isnull(),"Age"] = miss_mean
	print("dataframe1 first 40\n", dataframe1.head(30))

	#add new column UnderAge
	#set it to 1 if age less than 16 else 0
	dataframe1.insert(len(dataframe1.values[0]),"UnderAge",0)
	dataframe1.UnderAge = dataframe1.Age.apply(lambda x:0 if 15 <= x <= 1000 else 1)

	#insert new column for sheltered passengers who have cabins against their names
	#set it to 1 if cabin is assigned or else 0
	dataframe1.insert(len(dataframe1.values[0]) ,"Sheltered", 0)
	dataframe1.Sheltered = dataframe1.Cabin.notnull().astype(int)
	print (dataframe1.head(10))

	dataframe1.to_csv(csvtype)
	dataframe2=dataframe1.iloc[:,[0,2,4,5,8,9]]
	print("Features:\n",dataframe2.head(10))
	return (dataframe2)

def print_results(results):
	count=892
	print("PassengerId,Survived")
	for i in results:
		print(count,",",i,sep="")
		count += 1

training_csv  = pd.read_csv("train.csv",usecols=("Survived","Pclass","Name","Sex","Age","SibSp","Parch","Fare","Cabin"))
training_data = pd.DataFrame(training_csv,columns=["Pclass","Name","Sex","Age","SibSp","Parch","Fare","Cabin"])
features_training = generate_dataframe(training_data,"newtrain.csv")
labels_training = training_csv.Survived

test_csv = pd.read_csv("test.csv",usecols=["Pclass","Name","Sex","Age","SibSp","Parch","Fare","Cabin"])
#Quick hack for missing Fare, as we know the test data only contains one record without fare beloging to Pclass=3
test_csv.Fare.fillna(0,inplace=True)
features_test = generate_dataframe(test_csv,"newtest.csv")

from sklearn.svm import SVC

#clf = GaussianNB()
clf = SVC()
print("Inside Main now Features:\n",features_training.head(10))
print("Inside Main now Tests:\n",features_test.head(10))
print("Test Set:\n",features_test)
clf.fit(features_training,labels_training)
results = clf.predict(features_test)
print_results(results)
