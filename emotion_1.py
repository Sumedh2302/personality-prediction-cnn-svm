import tkinter as tk
 #from tkinter import *
from tkinter import messagebox as ms
import sqlite3
from keras.models import load_model
from PIL import Image, ImageTk
import re
import random
import os, os.path
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def upload():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.load_weights('model.h5')
     # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
        # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = { 2: "Conscientiousness", 3: "Openness", 4: "Agreeableness", 5: "Neuroticism", 6: "Extraversion"}
    sampleNum = 0
        # start the webcam feed
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    while True:
            # Find haar cascade to draw bounding box around face
        ret, img = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            print(maxindex)
            if maxindex == 2 :
                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/Conscientiousness/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                   # cv2.putText(img, 'Fearful', (x + w, y + h), 1, 1, (255, 0,0), 1)
                    #cv2.waitKey(100)
                    cv2.imshow('frame', img)
        ###############################################################################################################            #cv2.waitKey(1);
            # elif maxindex == 1:
            #         sampleNum = sampleNum + 1
            #         cv2.imwrite("dataset/Disgusted/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            #         #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #         #cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
            #         #cv2.waitKey(100)
            #         cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
                 ###############################################################################################################            #cv2.waitKey(1);
            # elif maxindex == 0:
            #         sampleNum = sampleNum + 1
            #         cv2.imwrite("dataset/angry/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            #        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #        # cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
            #         #cv2.waitKey(100)
            #         cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
                
         ###############################################################################################################            #cv2.waitKey(1);
            elif maxindex == 3:
                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/Openness/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                   # cv2.putText(img, 'happy', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    #cv2.waitKey(100)
                    cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
     
            elif maxindex == 4:
                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/Agreeableness/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                   # cv2.putText(img, 'neutral', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    #cv2.waitKey(100)
                    cv2.imshow('frame', img)
        ###########################################################################################################        #cv2.waitKey(1);
            elif maxindex == 5:
                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/Neuroticism/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                   # cv2.putText(img, 'sad', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    #cv2.waitKey(100)
                    cv2.imshow('frame', img)
                    # cv2.waitKey(1);
        ###########################################################################################################
     ###########################################################################################################        #cv2.waitKey(1);
            elif maxindex == 6:
                    sampleNum = sampleNum + 1
                    cv2.imwrite("dataset/Extraversion/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                   # cv2.putText(img, 'sad', (x + w, y + h), 1, 1, (255, 0, 0), 1)
                    #cv2.waitKey(100)
                    cv2.imshow('frame', img)
                    # cv2.waitKey(1);
                    
            cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)
        cv2.imshow('Video', cv2.resize(img,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
def files_count():
    

    happy = 'dataset//Openness'

    number_of_Happy_files = len([item for item in os.listdir(happy) if os.path.isfile(os.path.join(happy, item))])
    #print (number_of_Happy_files)
    A = "Openness Person are = {0}".format(number_of_Happy_files)
    print(A)
    #return A

    fear = 'dataset//Conscientiousness'
    number_of_Fear_files = len([item for item in os.listdir(fear) if os.path.isfile(os.path.join(fear, item))])
    #print(number_of_Fear_files)
    B = "Conscientiousness Person are = {0}".format(number_of_Fear_files)
    print(B)
    #return B

    sad = 'dataset//Neuroticism'
    number_of_sad_files = len([item for item in os.listdir(sad) if os.path.isfile(os.path.join(sad, item))])
    #print(number_of_sad_files)
    C = "Neuroticism Person are = {0}".format(number_of_sad_files)
    print(C)
    #return C

    neutral = 'dataset//Agreeableness'
    number_of_neutral_files = len([item for item in os.listdir(neutral) if os.path.isfile(os.path.join(neutral, item))])
    #print(number_of_neutral_files)
    D = "Agreeableness Person are = {0}".format(number_of_neutral_files)
    print(D)
    #return D

    Surprised = 'dataset//Extraversion'
    number_of_Surprised_files = len([item for item in os.listdir(Surprised) if os.path.isfile(os.path.join(Surprised, item))])
    #print(number_of_neutral_files)
    E = "Extraversion Person are = {0}".format(number_of_Surprised_files)
    print(E)
    #return E
  
    # Disgusted = 'dataset//Disgusted'
    # number_of_Disgusted_files = len([item for item in os.listdir(neutral) if os.path.isfile(os.path.join(neutral, item))])
    # #print(number_of_neutral_files)
    # F = "Disgusted Person are = {0}".format(number_of_neutral_files)
    # print(F)
    #return F
    
    from tkinter import messagebox as ms
    if int(number_of_Happy_files) > int(number_of_Fear_files) and int(number_of_Happy_files) > int(number_of_sad_files) and int(number_of_Happy_files) > int(number_of_neutral_files) and int(number_of_Happy_files) > int(number_of_Surprised_files):
        #str_label="Depression Evaluation is = 75% and Person Was Excellent "
        ms.showinfo("Message", "Person is Openness MOOD")
        str_label = "Person is Openness"
        print(str_label)
       

    elif int(number_of_Fear_files) > int(number_of_Happy_files) and int(number_of_Fear_files) > int(number_of_sad_files) and int(number_of_Fear_files) > int(number_of_neutral_files) and int(number_of_Fear_files) > int(number_of_Surprised_files):
        ms.showinfo("Message", "Person is Conscientiousness MOOD")
        str_label = "Person is Conscientiousness"
        print(str_label)

    elif int(number_of_neutral_files) > int(number_of_Happy_files) and int(number_of_neutral_files) > int(number_of_sad_files) and int(number_of_neutral_files) > int(number_of_Fear_files) and int(number_of_neutral_files) > int(number_of_Surprised_files):
        ms.showinfo("Message", "Person is Agreeableness MOOD")
        str_label = "Person is Agreeableness"
        print(str_label)
        
    elif int(number_of_sad_files) > int(number_of_Happy_files) and int(number_of_sad_files) > int(number_of_neutral_files) and int(number_of_sad_files) > int(number_of_Fear_files) and int(number_of_sad_files) > int(number_of_Surprised_files) :
        ms.showinfo("Message", "Person is Neuroticism MOOD")
        str_label = "Person is Neuroticism"
        print(str_label)
        
    elif int(number_of_Surprised_files) > int(number_of_Happy_files) and int(number_of_Surprised_files) > int(number_of_sad_files) and int(number_of_Surprised_files) > int(number_of_Fear_files) and int(number_of_Surprised_files) > int(number_of_neutral_files):
         ms.showinfo("Message", "Person is Extraversion MOOD")
         str_label = "Person is Extraversion"
         print(str_label)      
                
   
    return str_label