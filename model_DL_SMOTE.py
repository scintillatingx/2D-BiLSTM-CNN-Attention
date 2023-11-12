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
import jsonlines
np.random.seed(13)
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense, Activation, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau


# In[2]:


#读取数据，统计样本数量
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
#f = pd.read_csv('test_all.csv')
veri = np.array(f.loc[:,:])
x_veri,y_veri = veri[:,:46],veri[:,-1:] 

# x_veri1 = std_x.fit_transform(x_veri)
# x_veri2 = Normalizer(norm='l2').fit_transform(x_veri1)


# In[4]:


#构建模型，二维卷积
def build_CNN(fea_cnt,numb):
    inputs = Input(shape=(fea_cnt,), dtype='float32')
    embds = Dense(1024 ,activation='relu')(inputs)
    embds = Reshape((32,32,1))(embds)
    embds = Conv2D(1,(17,17),strides=(1,1),padding="valid")(embds)
#    embds = MaxPooling2D(pool_size=(4,4), strides=None, padding='valid')(embds)
    embds = Conv2D(1,(8,8),strides=(1,1),padding="valid")(embds)
    embds = MaxPooling2D(pool_size=(4,4), strides=None, padding='valid')(embds)
    embds = Flatten()(embds)
#    embds = Dropout(0.2)(embds)
    embds = Dense(64,activation='relu')(embds)
#    embds = Flatten()(embds)
    outputs = Dense(numb,activation='softmax')(embds)
    embds = Dropout(0.2)(embds)
    model = Model(inputs=inputs, outputs=outputs)
    adam =tf.keras.optimizers.Adam(0.001)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model


# In[5]:


CNN = build_CNN(fea_cnt,numb)
CNN.summary()


# In[6]:


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
    CNN.fit(x_train, y_train, batch_size = 64, epochs = 100, verbose=1)
    CNN.save('./CNN46.h5')
    y_train_pred = CNN.predict(x_test)
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


# ------------------------------------------- load model and test
CNN_test = tf.keras.models.load_model('./CNN46.h5')
CNN_pred = CNN_test.predict(x_veri, batch_size=64, verbose=1)
CNN_true = np.array(y_veri)

CNN_pred = np.array(CNN_pred)
CNN_pred1 = []
for i in CNN_pred:
    if i[0]>i[1] and i[0]>i[2] and i[0]>i[3] and i[0]>i[4]:
        CNN_pred1.append(0)
    elif i[1]>i[0] and i[1]>i[2] and i[1]>i[3] and i[1]>i[4]:
        CNN_pred1.append(1)
    elif i[2]>i[0] and i[2]>i[1] and i[2]>i[3] and i[2]>i[4]:
        CNN_pred1.append(2)
    elif i[3]>i[0] and i[3]>i[1] and i[3]>i[2] and i[3]>i[4]:
        CNN_pred1.append(3)
    else:
        CNN_pred1.append(4)
print(classification_report(CNN_true , CNN_pred1, digits = 4))


# In[8]:


cm = confusion_matrix(CNN_true , CNN_pred1)
print(cm)


# In[9]:


def BiLSTM_SMOTE():
    TIME_STEPS = 2
    INPUT_SIZE = 23
    model = Sequential()
    model.add(Bidirectional(LSTM(units=32, batch_input_shape=(None,TIME_STEPS, INPUT_SIZE),     
        return_sequences=True,),merge_mode='concat'))
    model.add(Dropout(0.2))
# # add output layer
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax')) 
    adam =tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer = adam,loss='binary_crossentropy',metrics=['accuracy']) #使用rmsprop或者adam，而不是sgd

    file = pd.read_csv('C:/Users/chengxin/Desktop/train46.csv')
    dataset = np.array(file.loc[:,:])
    train_label = np.array(dataset[:,-1:])
    print(train_label.shape)

    onehot_encoder = OneHotEncoder(sparse=False)
    train_label = np.array(train_label).reshape(len(train_label), 1)
    train_label = onehot_encoder.fit_transform(train_label)
   
    train = np.array(dataset[:,:46]).astype(np.float)
    std_x = StandardScaler()
    train = std_x.fit_transform(train)
    train = Normalizer(norm='l2').fit_transform(train)

    x_train,x_test,y_train,y_test =train_test_split(train, train_label, train_size=0.8,test_size=0.2,shuffle=True) #sklearn.model_selection.
    smo = SMOTE(random_state=66)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    x_train = x_train.reshape(-1,2,23)
    x_test = x_test.reshape(-1,2,23)
    pro_x = np.array(x_train)
    pro_px = np.array(x_test)
    
    pro_y = np.array(y_train)
    pro_py = np.array(y_test)
    print(pro_y.shape)

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
        
        model.fit(x_train, y_train, epochs=100, batch_size=64,verbose=1)
        model.save('./BiLSTM46.h5')

        y_train_pred = model.predict(x_test)
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

        pred_y.extend(y_train_pred)
        pro_y1.extend(y_test)

        p = []    
        for j in pro_y1:
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
    model.summary()


# In[10]:


BiLSTM_SMOTE()


# In[11]:


# ------------------------------------------- load model and test
x_veri = x_veri.reshape(-1,2,23)
BiLSTM_test = tf.keras.models.load_model('./BiLSTM46.h5')
BiLSTM_pred = BiLSTM_test.predict(x_veri, batch_size=64, verbose=1)
BiLSTM_true = np.array(y_veri)
BiLSTM_pred = np.array(BiLSTM_pred)
BiLSTM_pred1 = []
for i in BiLSTM_pred:
    if i[0]>i[1] and i[0]>i[2] and i[0]>i[3] and i[0]>i[4]:
        BiLSTM_pred1.append(0)
    elif i[1]>i[0] and i[1]>i[2] and i[1]>i[3] and i[1]>i[4]:
        BiLSTM_pred1.append(1)
    elif i[2]>i[0] and i[2]>i[1] and i[2]>i[3] and i[2]>i[4]:
        BiLSTM_pred1.append(2)
    elif i[3]>i[0] and i[3]>i[1] and i[3]>i[2] and i[3]>i[4]:
        BiLSTM_pred1.append(3)
    else:
        BiLSTM_pred1.append(4)
print(classification_report(BiLSTM_true , BiLSTM_pred1, digits = 4))


# In[12]:


cm = confusion_matrix(BiLSTM_true , BiLSTM_pred1)
print(cm)



# In[13]:


def BiGRU_SMOTE():
    TIME_STEPS = 2
    INPUT_SIZE = 23
    model = Sequential()
    model.add(Bidirectional(GRU(units=32, batch_input_shape=(None,TIME_STEPS, INPUT_SIZE),     
        return_sequences=True,),merge_mode='concat'))
    #model.add(Dropout(0.2))
# # add output layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh')) 
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5))
    model.add(Activation('softmax')) 
    adam =tf.keras.optimizers.Adam(0.005)
    model.compile(optimizer = adam,loss='binary_crossentropy',metrics=['accuracy']) #使用rmsprop或者adam，而不是sgd

    file = pd.read_csv('C:/Users/chengxin/Desktop/train46.csv')
    dataset = np.array(file.loc[:,:])
    train_label = np.array(dataset[:,-1:])
    print(train_label.shape)

    onehot_encoder = OneHotEncoder(sparse=False)
    train_label = np.array(train_label).reshape(len(train_label), 1)
    train_label = onehot_encoder.fit_transform(train_label)
   
    train = np.array(dataset[:,:46]).astype(np.float)
    std_x = StandardScaler()
    train = std_x.fit_transform(train)
    train = Normalizer(norm='l2').fit_transform(train)

    x_train,x_test,y_train,y_test =train_test_split(train, train_label, train_size=0.8,test_size=0.2,shuffle=True) #sklearn.model_selection.
    smo = SMOTE(random_state=111)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    x_train = x_train.reshape(-1,2,23)
    x_test = x_test.reshape(-1,2,23)
    pro_x = np.array(x_train)
    pro_px = np.array(x_test)
    
    pro_y = np.array(y_train)
    pro_py = np.array(y_test)
    print(pro_y.shape)

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
        
        model.fit(x_train, y_train, epochs=100, batch_size=64,verbose=1)
        model.save('./BiGRU46.h5')

        y_train_pred = model.predict(x_test)
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

        pred_y.extend(y_train_pred)
        pro_y1.extend(y_test)

        p = []    
        for j in pro_y1:
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
    model.summary()


# In[14]:


BiGRU_SMOTE()


# In[15]:


# ------------------------------------------- load model and test
x_veri = x_veri.reshape(-1,2,23)
BiGRU_test = tf.keras.models.load_model('./BiGRU46.h5')
BiGRU_pred = BiGRU_test.predict(x_veri, batch_size=64, verbose=1)
BiGRU_true = np.array(y_veri)
BiGRU_pred = np.array(BiGRU_pred)
BiGRU_pred1 = []
for i in BiGRU_pred:
    if i[0]>i[1] and i[0]>i[2] and i[0]>i[3] and i[0]>i[4]:
        BiGRU_pred1.append(0)
    elif i[1]>i[0] and i[1]>i[2] and i[1]>i[3] and i[1]>i[4]:
        BiGRU_pred1.append(1)
    elif i[2]>i[0] and i[2]>i[1] and i[2]>i[3] and i[2]>i[4]:
        BiGRU_pred1.append(2)
    elif i[3]>i[0] and i[3]>i[1] and i[3]>i[2] and i[3]>i[4]:
        BiGRU_pred1.append(3)
    else:
        BiGRU_pred1.append(4)
print(classification_report(BiGRU_true , BiGRU_pred1, digits = 4))


# In[16]:


cm = confusion_matrix(BiGRU_true, BiGRU_pred1)
print(cm)