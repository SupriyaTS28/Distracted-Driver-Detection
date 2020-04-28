import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import backend as K
from math import ceil
from PIL import Image as pil_image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras import optimizers


driver_list = pd.read_csv('D:/vkandukuri_d/Resting_Calmly/wayne_state/deep_learning/dataset/driver_imgs_list.csv')
train_image =[]
image_label = []

for ci in range(10):
    print(ci)
    filepath =  'D:/vkandukuri_d/Resting_Calmly/wayne_state/deep_learning/dataset/imgs/train/'

    imgs = os.listdir(filepath +'c' +str(ci))

    for i in range(len(imgs)):
        img = load_img(filepath+'c' +str(ci) + '/' + imgs[i],target_size=(150,150))
        image_arr1 = img_to_array(img)
        image_arr1 = np.expand_dims(image_arr1, axis=0)
        image_arr1 /= 255
        label = ci
        driver = driver_list[driver_list['img'] == imgs[i]]['subject'].values[0]
        train_image.append([image_arr1,label,driver])
        image_label.append(i)


## Randomly shuffling the images

import random
random.shuffle(train_image)

list_drivers = list(set(driver_list['subject']))
driv_nums = [2, 10, 13, 6, 19, 24]
driv_selected = []
for i in driv_nums:
    driv_selected.append(list_drivers[i])
    
x_val = []
y_val =[]
x_train = []
y_train =[]
for features,labels,drivers in train_image:
    if drivers in driv_selected:
        x_val.append(features)
        y_val.append(labels)
    else:
        x_train.append(features)
        y_train.append(labels)

print(len(x_train),len(x_val))
 

#K.tensorflow_backend._get_available_gpus()

dataset_path = 'D:/vkandukuri_d/Resting_Calmly/wayne_state/deep_learning/dataset/imgs/'

x_train = np.array(x_train).reshape(-1,150,150,3)
x_val = np.array(x_val).reshape(-1,150,150,3)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)



from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import Model
from keras.layers import Flatten,Dense,Input,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization

resnet_input = Input(shape=(150,150,3),name='Image_input')
model_resnet50_conv = ResNet50(weights='imagenet',include_top=False, input_shape=(150,150,3))
#model generation
output_resnet50 = model_resnet50_conv(resnet_input)
x=GlobalAveragePooling2D()(output_resnet50)
x=Dense(1024,activation='relu')(x) #dense1
x = Dropout(0.5)(x)
x=Dense(1024,activation='relu')(x) #dense1
x=BatchNormalization()(x)
#x = Dropout(0.5)(x)
x=Dense(512,activation='relu')(x) #dense2
x = Dense(10, activation='softmax', name='predictions')(x)


resnet_pretrain = Model(input = resnet_input , output = x)
#sgd_opt = optimizers.SGD(lr=0.001)
resnet_pretrain.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
## Augment data in train and validation dataset using ImageDataGEnerator
train_augm_datagenerator = ImageDataGenerator(rotation_range = 15,width_shift_range = 0.2,
                                              height_shift_range = 0.2, rescale = 1./255,
                                              shear_range=0.2,horizontal_flip=False,fill_mode='nearest')


test_augm_datagenerator = ImageDataGenerator(rescale = 1./255)
val_augm_datagenerator = ImageDataGenerator()

generator_train = train_augm_datagenerator.flow(x_train,y_train,batch_size = 64)
steps_train = ceil(len(x_train)//64)
steps_val = ceil(len(x_val)//64)

def scheduler(epoch):
  if epoch < 3:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (3 - epoch))

callback_lr = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint('model_resnet_added_layers_64batch.h5',monitor='val_loss',mode='max',save_best_only=True)
early_stopping = EarlyStopping(monitor='accuracy',patience=2,mode='max')

history = resnet_pretrain.fit(generator_train,epochs=5,steps_per_epoch=steps_train, validation_data = (x_val,y_val),
                              callbacks = [checkpoint,early_stopping],validation_steps=steps_val)

score,accuracy_val = model.evaluate(x_val,y_val,batch_size=64)
model.save("resnet50_save_model.h5")


test_image = []
i = 0
fig, ax = plt.subplots(1, 10, figsize = (50,50 )) 
for 
files = os.listdir('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset/test/test1')
nums = np.random.randint(low=1, high=len(files), size=10)
    
img = load_img('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset/test/test1/img_57.jpg'
                    ,target_size=(150,150))
image_arr1 = img_to_array(img)
print(image_arr1.shape)
image_arr1 = np.expand_dims(image_arr1, axis=0)
image_arr1 /= 255
pred = model.predict(image_arr1)
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']

decoded_predictions = dict(zip(class_labels, pred[0]))
pred_class = str(np.where(pred[i] == np.amax(pred[i]))[0][0])


