import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


driver_list = pd.read_csv('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/driver_imgs_list.csv')
train_image =[]
image_label = []




for ci in range(10):
    print(ci)
    filepath =  'C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/train/'

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
 
dataset_path = 'C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/'

x_train = np.array(x_train).reshape(-1,150,150,3)
x_val = np.array(x_val).reshape(-1,150,150,3)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation="relu"))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(10)) 
model.add(Activation("sigmoid"))

checkpoint = ModelCheckpoint('model2_adam_batch100.h5',monitor='val_loss',mode='max',save_best_only=True)
early_stopping = EarlyStopping(monitor='accuracy',patience=3,mode='max')
adam_opt = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy'])

## Augment data in train and validation dataset using ImageDataGEnerator
train_augm_datagenerator = ImageDataGenerator(rotation_range = 15,width_shift_range = 0.2,
                                              height_shift_range = 0.2, rescale = 1./255,
                                              shear_range=0.2,horizontal_flip=False,fill_mode='nearest')

test_augm_datagenerator = ImageDataGenerator(rescale = 1./255)
val_augm_datagenerator = ImageDataGenerator()
generator_train = train_augm_datagenerator.flow(x_train,y_train,batch_size =64)
steps_train = ceil(len(x_train)//64)
steps_val = ceil(len(x_val)//64)
history = model.fit(generator_train,epochs=20,steps_per_epoch=steps_train, validation_data = (x_val,y_val),
                              callbacks = [checkpoint,early_stopping],validation_steps=steps_val)

score,accuracy_val = model.evaluate(x_val,y_val,batch_size=64)
model.save("model2_adam_batch64_save_model1.h5")


test_image = []
i = 0
fig, ax = plt.subplots(1, 10, figsize = (50,50 )) 
for 
files = os.listdir('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset/test/test1')
nums = np.random.randint(low=1, high=len(files), size=10)
    
img = load_img('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset/test/test1/img_57.jpg'
                    ,target_size=(150,150))
image_arr1 = img_to_array(img)
#print(img.shape)
print(image_arr1.shape)
image_arr1 = np.expand_dims(image_arr1, axis=0)
image_arr1 /= 255
pred = model.predict(image_arr1)
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']

decoded_predictions = dict(zip(class_labels, pred[0]))
pred_class = str(np.where(pred[i] == np.amax(pred[i]))[0][0])


print('img_57.jpg was predicted as "' + str(class_labels[int(pred_class)]) + '"')

# getting one image per label of train dataset of kaggle
f, ax = plt.subplots(2,5, figsize = (150,150))
for i in range(10):
    label =[]
    for j in range(1):
        files = os.listdir('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/train/c' +str(i)+'/')
        img = load_img('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/train/c' +str(i)+'/' + str(files[i]))
        label.append(files[i])
        if i > 4:
            ax[1,i-5].imshow(img)
            ax[1,i-5].set_title('C'+str(i))
        else:
            ax[0,i].imshow(img)
            ax[0,i].set_title('C'+str(i))
        plt.show



files = os.listdir('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/train/c' +str(i)+'/')
img_1 = load_img('C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/train/c' +str(i)+'/' + str(files[i]))
img = img_to_array(img_1)
img = np.expand_dims(img, 0)
datagen =  ImageDataGenerator(rotation_range = 15,width_shift_range = 0.2,
                                          height_shift_range = 0.2,rescale = 1./255,
                                          shear_range=0.2,horizontal_flip=False,fill_mode='nearest')
datagen.fit(img)
for x, val in zip(datagen.flow(img, 
                save_to_dir='C:/Users/Manoj/Documents/wayne_state/SEM_2/deep learning/project/dataset-1/test_augdata',
         save_prefix='augumented',
        save_format='jpg'),range(2)) :  
    pass


        

