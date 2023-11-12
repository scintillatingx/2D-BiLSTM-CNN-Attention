#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import sys
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score #查准率
from sklearn.metrics import precision_score #查全率=召回率=敏感度
from sklearn.metrics import recall_score  #特异性：1-FPR
from sklearn.metrics import roc_curve  #f1值
from sklearn.metrics import f1_score  #Matthews相关系数
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import plot_importance
from numpy import sort
from sklearn.metrics import accuracy_score
import jsonlines
np.random.seed(13)
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense, Activation, Flatten, Attention
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Lambda, Dot, Concatenate
from tensorflow.keras.layers import Layer


# In[2]:


#读取数据，统计样本数量
#file = pd.read_csv('train_end.csv')
file = pd.read_csv('C:/Users/chengxin/Desktop/train46.csv')
#print(file.head())
#读取特征和对应标签,统计各类别样本的数量
dataset = np.array(file.loc[:,:])
x = np.array(dataset[:,:46])
y = np.array(dataset[:,-1:])
#数据标准化
std_x = StandardScaler()
x = std_x.fit_transform(x)
#数据正则化
x2 = Normalizer(norm='l2').fit_transform(x)

fea_cnt = len(x[0]) #特征个数
numb = max(y)+1 # 分类类别个数
print(fea_cnt)
print(numb)

# #标签0-1化
onehot_encoder = OneHotEncoder(sparse=False)
y = np.array(y).reshape(len(y), 1)
y = onehot_encoder.fit_transform(y)

#划分数据集
x_train,x_test,y_train,y_test =train_test_split(x2,y,train_size=0.8,test_size=0.2,shuffle=True)
smo = SMOTE(random_state=66)
x_train, y_train = SMOTE().fit_resample(x_train, y_train)


# In[3]:


f = pd.read_csv('C:/Users/chengxin/Desktop/test46.csv')
veri = np.array(f.loc[:,:])
x_veri,y_veri = veri[:,:46],veri[:,-1:] 

# x_veri1 = std_x.fit_transform(x_veri)
# x_veri2 = Normalizer(norm='l2').fit_transform(x_veri1)


# In[4]:


#构建模型，二维卷积
def CNN_BiLSTM():
    inputs = Input(shape=(2,23), dtype='float32')
    bilstm = Bidirectional(LSTM(units=128, activation='relu'))(inputs)
    embds = Dense(1024,activation='relu')(bilstm)
    embds = Reshape((32,32,1))(embds)
    embds = Conv2D(1,(16,16),strides=(2,2),padding="valid")(embds)
    embds = MaxPooling2D(pool_size=(4,4), strides=None, padding='valid')(embds)
    embds = Flatten()(embds)
    embds = Dense(64,activation='relu')(embds)
    attention_probs = Dense(64 ,activation='relu',name = 'attention_vec')(embds)
    attention_mul = Multiply()([embds, attention_probs])
    attention = Dense(128)(attention_mul)
    outputs = Dense(5,activation='softmax')(embds)
    embds = Dropout(0.2)(embds)
    model = Model(inputs=inputs, outputs=outputs)
    adam =tf.keras.optimizers.Adam(0.001)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model


# In[5]:


CNN_BiLSTM = CNN_BiLSTM()
CNN_BiLSTM.summary()


# In[6]:


x_train = x_train.reshape(-1,2,23)
x_test = x_test.reshape(-1,2,23)
# onehot_encoder = OneHotEncoder(sparse=False)
# y_train = np.array(y_train).reshape(len(y_train), 1)
# y_train = onehot_encoder.fit_transform(y_train)
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
    CNN_BiLSTM.fit(x_train, y_train, batch_size = 64, epochs = 100, verbose=1)
    CNN_BiLSTM.save('./CNN_BiLSTM46.h5')
    y_train_pred = CNN_BiLSTM.predict(x_test)
    pre_test_y = []
    for i in y_train_pred:
        if i[0]>i[1] and i[0]>i[2] and i[0]>i[3] and i[0]>i[4]:
            pre_test_y.append(0)
        elif i[1]>i[0] and i[1]>i[2] and i[1]>i[3] and i[1]>i[4]:
            pre_test_y.append(1)
        elif i[2]>i[0] and i[2]>i[1] and i[2]>i[3] and i[2]>i[4]:
            pre_test_y.append(2)
        elif i[3]>i[0] and i[3]>i[1] and i[3]>i[2] and i[3]>i[4]:
            pre_test_y.append(3)
        else:
            pre_test_y.append(4)

    Y_test1 = np.array(y_test)
    p = []
    for j in Y_test1:
        if j[0] == 1 and j[1] == 0 and j[2] == 0 and j[3] == 0 and j[4] == 0:
            p.append(0)
        elif j[0] == 0 and j[1] == 1 and j[2] == 0 and j[3] == 0 and j[4] == 0:
            p.append(1)
        elif j[0] == 0 and j[1] == 0 and j[2] == 1 and j[3] == 0 and j[4] == 0:
            p.append(2)
        elif j[0] == 0 and j[1] == 0 and j[2] == 0 and j[3] == 1 and j[4] == 0:
            p.append(3)
        else:
            p.append(4)
    print(classification_report(p , pre_test_y, digits = 4))


# In[7]:


x_veri = x_veri.reshape(-1,2,23)
CNN_BiLSTM_test = tf.keras.models.load_model('./CNN_BiLSTM46.h5')
CNN_BiLSTM_pred = CNN_BiLSTM_test.predict(x_veri, batch_size=64, verbose=1)
CNN_BiLSTM_true = np.array(y_veri)

CNN_BiLSTM_pred = np.array(CNN_BiLSTM_pred)
CNN_BiLSTM_pred1 = []
for i in CNN_BiLSTM_pred:
    if i[0]>i[1] and i[0]>i[2] and i[0]>i[3] and i[0]>i[4]:
        CNN_BiLSTM_pred1.append(0)
    elif i[1]>i[0] and i[1]>i[2] and i[1]>i[3] and i[1]>i[4]:
        CNN_BiLSTM_pred1.append(1)
    elif i[2]>i[0] and i[2]>i[1] and i[2]>i[3] and i[2]>i[4]:
        CNN_BiLSTM_pred1.append(2)
    elif i[3]>i[0] and i[3]>i[1] and i[3]>i[2] and i[3]>i[4]:
        CNN_BiLSTM_pred1.append(3)
    else:
        CNN_BiLSTM_pred1.append(4)
print(classification_report(CNN_BiLSTM_true , CNN_BiLSTM_pred1, digits = 4))


# In[8]:


cm = confusion_matrix(CNN_BiLSTM_true, CNN_BiLSTM_pred1)
print(cm)



