# Training a neural network for classifying images with and without mask

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models


train = ImageDataGenerator(rescale = 1/255) 
validation = ImageDataGenerator(rescale = 1/255)


train_dataset = train.flow_from_directory('G:/My Projects/computer-vision/COVID-19/train/',
                                          target_size = (200,200), batch_size = 3, class_mode = 'binary') # Here we have 2 classes (binary), with and without mask
validation_dataset = train.flow_from_directory('G:/My Projects/computer-vision/COVID-19/validation/',
                                          target_size = (200,200), batch_size = 3, class_mode = 'binary')

train_dataset.class_indices # Represents each class with a binary number (0 and 1)

# Network Architecture
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                                                        
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2), 
                                    
                                    tf.keras.layers.Flatten(),
                                    
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dense(1, activation = 'sigmoid' )])     
model.summary()      


model.compile(loss = 'binary_crossentropy', 
              optimizer = 'rmsprop', # Can also try with Adam
              metrics = ['accuracy'])
              
history = model.fit(train_dataset,
                      steps_per_epoch = 5, 
                      epochs = 10, # Epochs are the number of iterations your optimizer takes, you can use tensorflow callbacks to stop the training when reached certain accuracy
                      validation_data = validation_dataset)

print(model.evaluate(train_dataset))
print(model.evaluate(validation_dataset))

# Plotting Acuracy and Val_Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


face_clsfr=cv2.CascadeClassifier(r'C:\Users\ashwi\OneDrive\Desktop\ICS_Sylabus\Research_Project\Code\CNN codes\FaceDetectionWithMask\haarcascade_frontalface_default.xml')
dir_path = r"G:/My Projects/computer-vision projects/basedata/test" # Path to the images on which we predict

for i in os.listdir(dir_path):
    #img = image.load_img(dir_path + '//' + i, target_size=(200,200))
    img = cv2.imread(dir_path + '//' + str(i))
    img = cv2.resize(img,(200,200))
    
    X = image.img_to_array(img) # Always the Images should be converted to array before feeding to the neural network
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)        
    val = model.predict(images) # Outputs 0 or 1. with mask:0, without mask:1
    print(val) 
    
    if val== 0:        
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # R G B, Green Bounding box for images with mask
            cv2.putText(img,'With Mask',(x,y-10), font, 1, (200,255,155))
        
        plt.imshow(img)
        plt.show()
        print('Predicted : With Mask')
        
    else:
        
        for (x,y,w,h) in faces:            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # R G B, Red Bounding box for images without mask
            cv2.putText(img,'No Mask',(x,y-10), font, 1, (200,255,155))
        
        plt.imshow(img)
        plt.show()
        print('Predicted : Without Mask') 
