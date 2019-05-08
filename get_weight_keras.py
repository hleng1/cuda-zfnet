from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
a = model.get_weights()

# all layers
l = model.layers
#weights of 8 layers
layer1 = l[0].get_weights()#96
layer2 = l[2].get_weights()#256
layer3 = l[4].get_weights()#384
layer4 = l[5].get_weights()#384
layer5 = l[6].get_weights()#256
layer6 = l[9].get_weights()#4096
layer7 = l[11].get_weights()#4096
layer8 = l[13].get_weights()#1000

f=open('layer1.txt',"x")
f=open('bias1.txt',"x")
for a in layer1[0]:
    for b in a:
        for c in b:
            for d in c:
                f=open('layer1.txt',"a")
                f.write(str(d))
                f.write('\n')
                f.close()
for i in layer1[1]:
    f=open('bias1.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer2.txt',"x")
f=open('bias2.txt',"x")
for a in layer2[0]:
    for b in a:
        for c in b:
            for d in c:
                f=open('layer2.txt',"a")
                f.write(str(d))
                f.write('\n')
                f.close()
for i in layer2[1]:
    f=open('bias2.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer3.txt',"x")
f=open('bias3.txt',"x")
for a in layer3[0]:
    for b in a:
        for c in b:
            for d in c:
                f=open('layer3.txt',"a")
                f.write(str(d))
                f.write('\n')
                f.close()
for i in layer3[1]:
    f=open('bias3.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer4.txt',"x")
f=open('bias4.txt',"x")
for a in layer4[0]:
    for b in a:
        for c in b:
            for d in c:
                f=open('layer4.txt',"a")
                f.write(str(d))
                f.write('\n')
                f.close()
for i in layer4[1]:
    f=open('bias4.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer5.txt',"x")
f=open('bias5.txt',"x")
for a in layer5[0]:
    for b in a:
        for c in b:
            for d in c:
                f=open('layer5.txt',"a")
                f.write(str(d))
                f.write('\n')
                f.close()
for i in layer5[1]:
    f=open('bias5.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer6.txt',"x")
f=open('bias6.txt',"x")
for a in layer6[0]:
    for b in a:
        f=open('layer6.txt',"a")
        f.write(str(b))
        f.write('\n')
        f.close()
for i in layer1[1]:
    f=open('bias6.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer7.txt',"x")
f=open('bias7.txt',"x")
for a in layer7[0]:
    for b in a:
        f=open('layer7.txt',"a")
        f.write(str(b))
        f.write('\n')
        f.close()
for i in layer7[1]:
    f=open('bias7.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()

f=open('layer8.txt',"x")
f=open('bias8.txt',"x")
for a in layer8[0]:
    for b in a:
        f=open('layer8.txt',"a")
        f.write(str(b))
        f.write('\n')
        f.close()
for i in layer8[1]:
    f=open('bias8.txt',"a")
    f.write(str(i))
    f.write('\n')
    f.close()
