import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import keras

# ### loading mist hand written dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)




# ## Applying threshold for removing noise 

_,X_train_th = cv2.threshold(X_train,127,255,cv2.THRESH_BINARY)
_,X_test_th = cv2.threshold(X_test,127,255,cv2.THRESH_BINARY)



# ### Reshaping 

X_train = X_train_th.reshape(-1,28,28,1)
X_test = X_test_th.reshape(-1,28,28,1)


# ### Creating categorical output from 0 to 9

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# ## cross checking shape of input and output

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


input_shape = (28,28,1)
num_of_class = 10
model = Sequential()#顺序模型是多个网络层的线性堆叠
#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#filters - 输出空间维度
#kernal size - 卷积核大小

model.add(Conv2D(32,kernel_size=(3,3),activation = 'relu',input_shape = input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
#keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(MaxPool2D(pool_size=(2,2)))
# 0.25 为 25%概率被drop了
model.add(Dropout(0.25))
#将输入展平。不影响批量大小。为了接下来的dense
model.add(Flatten())
#units: 正整数，输出空间维度。
#
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
#概率输出
model.add(Dense(num_of_class,activation='softmax'))


loss = keras.losses.categorical_crossentropy
op = keras.optimizers.Adadelta()

model.compile(loss=loss,optimizer=op,metrics=['accuracy'])
model.summary()

history = model.fit(X_train,y_train,epochs=5,shuffle=True,batch_size=128,validation_data=(X_test,y_test))
#save the model， for further use
model.save('digit_classifier.h5')
