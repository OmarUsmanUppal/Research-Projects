import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

'''=========READ FILE================'''

car_data = pd.read_csv("Datasets-UCI//6_car.csv", delimiter=';')

'''SPLIT THE DATA BETWEEN ATTRIBUTES and LABEL variables'''
num_of_attr = car_data.shape[1]-1
num_of_samples = car_data.shape[0]
class_types = car_data["label"].unique()
class_size = len(class_types)

X_attr = np.array(car_data.iloc[:, 0:num_of_attr])
y_label = np.array(car_data['label'])

'''SAMPLES SPLIT BETWEEN TRAIN and TEST DATA'''

print("Experiment of BernoulliNB using Selected Sample")
train_size = int(num_of_samples * 0.8)
test_size = num_of_samples - train_size

x_train = X_attr[:train_size, :]
x_test = X_attr[train_size:, :]

y_train = y_label[:train_size]
y_test = y_label[train_size:]

# EXPERIMENT 2 Using BernoulliNB with the same data set we used in First Experiement of NBC forumula
print("="*60)
clf1 = BernoulliNB()
clf1.fit(x_train, y_train)
y_predicted1 = clf1.predict(x_train)
print("Accuracy of Train Data using selected sample is {}%".format(round(accuracy_score(y_train, y_predicted1)*100), 2))
print("-"*60)
clf1 = BernoulliNB()
clf1.fit(x_train, y_train)
y_predicted1 = clf1.predict(x_test)
print("Accuracy of Test Data using selected sample is {}%".format(round(accuracy_score(y_test, y_predicted1)*100), 2))
print("="*60, "\n")

# EXPERIMENT 3 Using BernoulliNB with Random Samples
print("="*60)
print("Experiment of Bernoulli using Random Sample")
print("-"*60)
x_train, x_test, y_train, y_test = train_test_split(X_attr, y_label, test_size=0.2, random_state=42)

clf1.fit(x_train, y_train)
y_predicted1 = clf1.predict(x_train)
print("Accuracy of Train Data using random sample is {} %".format(round(accuracy_score(y_train, y_predicted1)*100), 2))
print("-"*60)
clf1 = BernoulliNB()
clf1.fit(x_train, y_train)
y_predicted1 = clf1.predict(x_test)
print("Accuracy of Test Data using random sample is {} %".format(round(accuracy_score(y_test, y_predicted1)*100), 2))
print("="*60, "\n")

print("="*60)


