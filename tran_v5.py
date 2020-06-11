import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import keras as K
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input,BatchNormalization, Convolution2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam, SGD,Adadelta
from keras.applications.mobilenet import MobileNet

num_class=79
base_model = MobileNet(include_top=False, input_shape=(224,224,3),weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
# x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x)

preds=Dense(num_class, activation='softmax')(x) 

model=Model(inputs=base_model.input,outputs=preds)
model.load_weights(".\manh\modelv5-022.hdf5")
model.summary()


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,brightness_range=[0.2,1.0])
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory('./dataset_area/',
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                  
                                                 class_mode='sparse',
                                                 shuffle=True
                                                
                                                 )


validation_generator = test_datagen.flow_from_directory(
                                                './dataset_area_test/', # same directory as training data
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='sparse',
                                                shuffle=True
                                                
                                                )
np.save('tuoi', train_generator.class_indices)
epochs = 50
learning_rate = 0.001
decay_rate = learning_rate / epochs
opt = SGD(lr=learning_rate, decay=decay_rate)
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adadelta(),
              metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint('modelv5-{epoch:03d}.hdf5',
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')
callbacks_list = [checkpoint]
step_size_train = train_generator.n/train_generator.batch_size
step_size_val = validation_generator.samples // validation_generator.batch_size
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator, 
                   validation_steps = step_size_val,
                   callbacks = callbacks_list,
                   epochs=epochs)