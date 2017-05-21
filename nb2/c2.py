import numpy as np

#features_train = np.loadtxt("iris_training.csv",delimiter=",",usecols=(0,1,2,3),skiprows=1)

features_train = np.loadtxt("iris_training.csv",delimiter=",",skiprows=1,usecols=(0,1,2,3))
labels_train = np.loadtxt("iris_training.csv",delimiter=",",skiprows=1,usecols=4)

features_test = np.loadtxt("iris_test.csv",delimiter=",",skiprows=1,usecols=(0,1,2,3))
labels_test = np.loadtxt("iris_test.csv",delimiter=",",skiprows=1,usecols=4)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(features_train,labels_train)

labels_predict = classifier.predict(features_test)

from sklearn.metrics import accuracy_score

a_score = accuracy_score(labels_predict,labels_test)

print("Accuracy Score:",a_score)
