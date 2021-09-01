# -*- coding: utf-8 -*-

from  sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
dataset=load_breast_cancer()
data=dataset['data']
# print(data)
target=dataset['target']
# print(target)

train_data,test_data,train_target,test_target=train_test_split(data,target)
# print(train_data)
# print(test_data)
# print(train_target)
# print(test_target)

classifier=GaussianNB()
classifier.fit(train_data,train_target)
predicts=classifier.predict(test_data)

print(predicts)

isabet=accuracy_score(test_target,predicts)
print('İsabetlik:',isabet)