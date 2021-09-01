# -*- coding: utf-8 -*-

from  sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


dataset=load_breast_cancer()
data=dataset['data']
target=dataset['target']
train_data,test_data,train_target,test_target=train_test_split(data,target)

scaler=StandardScaler()
scaler.fit(train_data)

normalize_train=scaler.transform(train_data)
normalize_test=scaler.transform(test_data)


# # Tek Katmanlı
# classifier=Perceptron()

# Çok Katmanlı
classifier=MLPClassifier(hidden_layer_sizes=(50,50,50))
classifier.fit(normalize_train,train_target)

y_predictions=classifier.predict(normalize_test)

print('Confusion Matrix:')
print(confusion_matrix(y_predictions,test_target))