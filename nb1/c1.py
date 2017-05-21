import numpy as np

features_train_np = np.loadtxt("c1-train-features.csv",delimiter=",",usecols=(1,3),skiprows=1)
labels_train_np = np.loadtxt("c1-train-labels.csv",delimiter=",",usecols=0,skiprows=1)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(features_train_np,labels_train_np)

