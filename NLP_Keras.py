
# coding: utf-8

# In[12]:


import keras
import numpy as np
from keras.datasets import imdb


# In[13]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[29]:


(x_train,y_train),(x_test,y_test) = imdb.load_data()
#x_train[0]


# In[30]:


print(x_train.shape,y_train.shape)


# In[31]:


m = max(list(map(len,x_train)),list(map(len,x_test)))
#print(m)


#     文本处理，计算最长文本长度，筛出特殊值，padding对齐

# In[32]:


maxword= 400
x_train = sequence.pad_sequences(x_train,maxlen = maxword)
x_test = sequence.pad_sequences(x_test,maxlen = maxword)
vocab_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1
print(vocab_size)


#     网络搭建

# In[33]:


model = Sequential()
#embeding layer
model.add(Embedding(vocab_size,64,input_length=maxword))
#vectorization
model.add(Flatten())

#full connected layer
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
          
          
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[34]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=256,verbose=1)
score = model.evaluate(x_train,y_train)


# In[ ]:


#RNN
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(vocab_size,64,input_length=maxword))
model.add(LSTM(128,return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dropout(0.25))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print(model.summary())


#CNN
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling1D

model = Sequential()
model.add(Embedding(vocab_size,64,input_length=maxword))
#卷积层
model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
#展平
model.add(Flatten())

#全连接
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=512)
scoreCNN = model.evaluate(x_test,y_test,verbose=1)
print(scoreCNN)