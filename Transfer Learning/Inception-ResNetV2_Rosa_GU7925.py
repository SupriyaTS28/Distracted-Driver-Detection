#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


import keras
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing import image 
from keras.layers.normalization import BatchNormalization
from keras import optimizers

from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# In[2]:


driver_details = pd.read_csv('driver_imgs_list.csv', na_values = 'na')
driver_details.head()


# In[3]:


train_image = []
image_label = []

for i in range(10):
    print('now in folder C',i)
    imgs = os.listdir('imgs/train/c'+str(i))
    for j in range(len(imgs)):
        img_name = 'imgs/train/c'+str(i)+'/'+imgs[j]
        img = cv2.imread(img_name)
        img = img[50:,120:-50]
        img = cv2.resize(img,(299,299))
        label = i
        driver = driver_details[driver_details['img'] == imgs[j]]['subject'].values[0]
        train_image.append([img,label,driver])
        image_label.append(i)


# In[4]:


## Randomly shuffling the images

import random
random.shuffle(train_image)


# In[5]:


D = []
for features,labels,drivers in train_image:
    D.append(drivers)

## Deduplicating drivers

deduped = []

for i in D:
    if i not in deduped:
        deduped.append(i)
    

## selecting random drivers for the validation set
driv_selected = []
import random
driv_nums = random.sample(range(len(deduped)), 5)
for i in driv_nums:
    driv_selected.append(deduped[i])


# In[6]:


## Splitting the train and test

X_train= []
y_train = []
X_test = []
y_test = []
D_train = []
D_test = []

for features,labels,drivers in train_image:
    if drivers in driv_selected:
        X_test.append(features)
        y_test.append(labels)
        D_test.append(drivers)
    
    else:
        X_train.append(features)
        y_train.append(labels)
        D_train.append(drivers)
    
print (len(X_train),len(X_test))
print (len(y_train),len(y_test))


# In[7]:


## Converting images to nparray. Encoding the Y

X_train = np.array(X_train).reshape(-1,299,299,3)
X_test = np.array(X_test).reshape(-1,299,299,3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print (X_train.shape)


# In[8]:


inception_resnet_input = Input(shape = (299,299,3),name = 'Image_input')

base_model = InceptionResNetV2(weights='imagenet',include_top=False,input_tensor = inception_resnet_input)
base_model.summary()


# In[9]:


output_inception = base_model(inception_resnet_input)
#Add the fully-connected layers 
x=GlobalAveragePooling2D()(output_inception)
x=Dense(1024,activation='relu')(x) #dense layer 1
x = Dropout(0.1)(x) 
x=Dense(1024,activation='relu')(x) #dense layer 2
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x) #dense layer 3
x = Dense(10, activation='softmax', name='predictions')(x)

inception_resnet = Model(input = inception_resnet_input, output = x)
inception_resnet.summary()


# In[10]:


#Freeze the base layers
for layer in base_model.layers:
    layer.trainable = False
for layer in inception_resnet.layers:
    print(layer, layer.trainable)


# In[11]:


sgd = optimizers.SGD(lr=0.001)
inception_resnet.compile(loss='categorical_crossentropy',optimizer = sgd, metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint,EarlyStopping

checkpointer = ModelCheckpoint('inception_resnet_weights_sgd.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = 3, verbose=1)


train_gen = ImageDataGenerator(rescale=1./255,
                               height_shift_range=0.2,
                               width_shift_range = 0.2,
                               zoom_range = 0.2,
                               rotation_range=40,
                               )

val_gen = ImageDataGenerator(rescale = 1./255, 
                             )

train_generator = train_gen.flow(X_train,y_train,batch_size=16)
val_generator = val_gen.flow(X_test,y_test, batch_size=16)


inception_resnet.fit_generator(train_generator, steps_per_epoch = train_generator.n//16, epochs=10, 
                               validation_data=val_generator, validation_steps = val_generator.n//16, 
                               callbacks=[checkpointer, earlystopper])


inception_resnet.evaluate_generator(val_generator,val_generator.n//16)




