import scipy.io
import ast 
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
import numpy as np
import cv2
from keras.models import Model, Sequential
# from mtcnn.mtcnn import MTCNN
num_class=79
base_model = MobileNet(include_top=False, input_shape=(224,224,3),weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
# x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x)

preds=Dense(num_class, activation='softmax')(x) 

model=Model(inputs=base_model.input,outputs=preds)
model.load_weights("./modelv5-016.hdf5")
face_cascade = cv2.CascadeClassifier(".\models\haarcascade_frontalface_default.xml")
# face_cascade = MTCNN()
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img




class_indices = np.load('./tuoi.npy', allow_pickle = True)
res = ast.literal_eval(str(class_indices))
def age(apparent_age):
	tam=""
	for x, y in res.items():
		if int(y) == apparent_age:
			print (x)	
			tam = str(x)
	return tam




cap = cv2.VideoCapture(0) 

while(True):
	ret, img = cap.read()
	
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: 
			
			
			
			cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1) 
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
			try:
				
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			try:
				
				detected_face = cv2.resize(detected_face, (224, 224))
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				
				age_distributions = model.predict(img_pixels)
				
				apparent_age = np.argmax(age_distributions)
				
				
				manh = age(apparent_age)
				print (manh)
				info_box_color = (32, 84, 216 )
				
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
				
				cv2.putText(img,manh, (x+int(w/2)-10, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
			except Exception as e:
				print("exception",str(e))
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
	
	
cap.release()
cv2.destroyAllWindows()
print (class_indices)