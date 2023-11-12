#!/usr/bin/env python
# coding: utf-8

# In[1]:


.5import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from collections import Counter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR, SVC, NuSVC
import xgboost
from numpy import sort
from xgboost import XGBClassifier
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel


# In[2]:


#读取训练集数据，统计样本数量
file = pd.read_csv('C:/Users/chengxin/Desktop/train46.csv')
print(file.shape)

#读取特征和对应标签,统计各类别样本的数量
dataset = np.array(file.loc[:,:])
X,y = dataset[:,:46],dataset[:,-1:] #x为数据，y为标签

#数据标准化
std_x = StandardScaler()
X1 = std_x.fit_transform(X)

#数据正则化
x = Normalizer(norm='l2').fit_transform(X1)      #'l1'\'l2'\'max'

#划分数据与标签
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8,test_size=0.2,shuffle=True)

# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=66)
x_train, y_train = SMOTE().fit_resample(x_train, y_train)


# In[3]:


f = pd.read_csv('C:/Users/chengxin/Desktop/test46.csv')
print(f.shape)
veri = np.array(f.loc[:,:])
x_veri,y_veri = veri[:,:46],veri[:,-1:] #x为数据，y为标签



# In[4]:


#SVM训练集
pro_x = np.array(x_train)
pro_px = np.array(x_test)  
pro_y = np.array(y_train)
pro_py = np.array(y_test)
print('10折正在进行......')
cv = KFold(n_splits=10, shuffle=True)
probass_y = []
NBtest_index = []
pred_y = []
pro_y1 = []
pre_test_y = []
for train, test in cv.split(pro_x):  # train test  是下标
    x_train, x_test = pro_x[train], pro_x[test]
    y_train, y_test = pro_y[train], pro_y[test]
    NBtest_index.extend(test)
    clf_svm = SVC(probability=True)
    clf_svm.fit(x_train,y_train)
    y_pred_svm = clf_svm.predict(x_test)
    score_svm=clf_svm.score(x_test,y_test)
    print(classification_report(y_test,y_pred_svm, digits = 4))


# In[5]:


#SVM测试集
y_veri_pred_svm = clf_svm.predict(x_veri)
y_veri_score_svm = clf_svm.score(x_veri, y_veri)
print(classification_report(y_veri,y_veri_pred_svm, digits = 4))


# In[6]:


cm = confusion_matrix(y_veri,y_veri_pred_svm)
print(cm)


# In[7]:


#RF训练集
pro_x = np.array(x_train)
pro_px = np.array(x_test)  
pro_y = np.array(y_train)
pro_py = np.array(y_test)         
print('10折正在进行......')
cv = KFold(n_splits=10, shuffle=True)
probass_y = []
NBtest_index = []
pred_y = []
pro_y1 = []
pre_test_y = []
for train, test in cv.split(pro_x):  # train test  是下标
    x_train, x_test = pro_x[train], pro_x[test]
    y_train, y_test = pro_y[train], pro_y[test]
    NBtest_index.extend(test)
    clf_rf = RandomForestClassifier(random_state = 66)
    clf_rf.fit(x_train,y_train)
    y_pred_rf = clf_rf.predict(x_test)
    score_rf = clf_rf.score(x_test,y_test)
    print(classification_report(y_test,y_pred_rf, digits = 4))


# In[8]:


y_veri_pred_rf = clf_rf.predict(x_veri)
y_veri_score_rf = clf_rf.score(x_veri, y_veri)
print(classification_report(y_veri,y_veri_pred_rf, digits = 4))


# In[9]:


#RF训练集
pro_x = np.array(x_train)
pro_px = np.array(x_test)  
pro_y = np.array(y_train)
pro_py = np.array(y_test)
print('10折正在进行......')
cv = KFold(n_splits=10, shuffle=True)
probass_y = []
NBtest_index = []
pred_y = []
pro_y1 = []
pre_test_y = []
for train, test in cv.split(pro_x):  # train test  是下标
    x_train, x_test = pro_x[train], pro_x[test]
    y_train, y_test = pro_y[train], pro_y[test]
    NBtest_index.extend(test)
    clf_rf = RandomForestClassifier(random_state = 1)
    clf_rf.fit(x_train,y_train)
    y_pred_rf = clf_rf.predict(x_test)
    score_rf = clf_rf.score(x_test,y_test)
    print(classification_report(y_test,y_pred_rf, digits = 4))


# In[10]:


y_veri_pred_rf = clf_rf.predict(x_veri)
y_veri_score_rf = clf_rf.score(x_veri, y_veri)
print(classification_report(y_veri,y_veri_pred_rf, digits = 4))


# In[11]:


cm = confusion_matrix(y_veri,y_veri_pred_rf)
print(cm)



# In[12]:


#XGBoost训练集
pro_x = np.array(x_train)
pro_px = np.array(x_test)  
pro_y = np.array(y_train)
pro_py = np.array(y_test)
print('10折正在进行......')
cv = KFold(n_splits=10, shuffle=True)
probass_y = []
NBtest_index = []
pred_y = []
pro_y1 = []
pre_test_y = []
for train, test in cv.split(pro_x):  # train test  是下标
    x_train, x_test = pro_x[train], pro_x[test]
    y_train, y_test = pro_y[train], pro_y[test]
    NBtest_index.extend(test)
    clf_xgb = XGBClassifier()
    clf_xgb.fit(x_train,y_train)
    y_pred_xgb = clf_xgb.predict(x_test)
    score_xgb=clf_xgb.score(x_test,y_test)
    print(classification_report(y_test,y_pred_xgb, digits = 4))


# In[13]:


y_veri_pred_xgb = clf_xgb.predict(x_veri)
y_veri_score_xgb = clf_xgb.score(x_veri,y_veri)
print(classification_report(y_veri,y_veri_pred_xgb, digits = 4))


# In[14]:


cm = confusion_matrix(y_veri,y_veri_pred_xgb)
print(cm)


