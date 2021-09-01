# -*- coding: utf-8 -*-
from  sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

dataset=load_iris()
data=dataset.data[:,0:2]
target=dataset.target
x_data=data[:,0]
y_data=data[:,1]

neighbor_sayisi=3
classifier=KNeighborsClassifier(n_neighbors=neighbor_sayisi)
classifier.fit(data,target)

x_min,x_max=x_data.min()-1,x_data.max()+1
y_min,y_max=y_data.min()-1,y_data.max()+1
h=0.02
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
zz=classifier.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

plt.pcolormesh(xx,yy,zz,cmap=plt.cm.Paired)
plt.scatter(x_data,y_data,c=target, cmap=plt.cm.Paired,edgecolor='k')