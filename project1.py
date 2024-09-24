# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:22:05 2023

@author: karthik
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


daisy_dir= os.path.join(r'C:\Users\karthik\OneDrive\Documents\flowers\daisy')
dandelion_dir= os.path.join(r'C:\Users\karthik\OneDrive\Documents\flowers\dandelion')
rose_dir= os.path.join(r'C:\Users\karthik\OneDrive\Documents\flowers\rose')
sunflower_dir= os.path.join(r'C:\Users\karthik\OneDrive\Documents\flowers\sunflower')
tulip_dir= os.path.join(r'C:\Users\karthik\OneDrive\Documents\flowers\tulip')
train_tulip_names=os.listdir(tulip_dir)
print(train_tulip_names[:5])
train_sunflower_names=os.listdir(sunflower_dir)
print(train_sunflower_names[:5])
batch_size=16
#it will divide images into different images

train_datagen=ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(r'C:\Users\karthik\OneDrive\Documents\flowers',
                                                    target_size=(200,200),
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    classes=['daisy','dandelion','rose','sunflower','tulip'],
                                                    class_mode='categorical')
target_size=(200,200)
model=tf.keras.models.Sequential([
    # note the input shape is desired size of the image 500*500 with 3 bytes color 
    #the first convolution
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=(200,200,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    #the second convolution 
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # the third 
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Flatten the results to feed into a dense layer 
    tf.keras.layers.Flatten(),
    #64 neuron in the fully -connected layer 
    tf.keras.layers.Dense(128, activation='relu'),
    #5 output neurons for 5 classes with the softmax activation 
    tf.keras.layers.Dense(5,activation='softmax')
    ])

                          
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
#Total sample count 
total_sample=train_generator.n
#training
num_epoches=5
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epoches,verbose=1)

model_json=model.to_json()
with open(r'C:\Users\karthik\OneDrive\Documents\Ai\modelGG.json',"w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights(r"C:\Users\karthik\OneDrive\Documents\Ai\model1GG.h5")
print("saved model to disk")    