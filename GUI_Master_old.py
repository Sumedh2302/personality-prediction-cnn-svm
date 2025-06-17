import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
#import CNNModel 
import sqlite3
# import svm as svm
from sklearn import svm
import pickle
from sklearn.preprocessing import LabelEncoder
from skimage import feature

# import decisiontree as dt

#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="#191970")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("HOMEPAGE")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('im2.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)

# img=ImageTk.PhotoImage(Image.open("l1.jpg"))

# img2=ImageTk.PhotoImage(Image.open("l2.jpg"))

# img3=ImageTk.PhotoImage(Image.open("l3.jpg"))


# logo_label=tk.Label()
# logo_label.place(x=0,y=0)

# x = 1

# # function to change to next image
# def move():
# 	global x
# 	if x == 4:
# 		x = 1
# 	if x == 1:
# 		logo_label.config(image=img)
# 	elif x == 2:
# 		logo_label.config(image=img2)
# 	elif x == 3:
# 		logo_label.config(image=img3)
# 	x = x+1
# 	root.after(2000, move)

# # calling the function
# move()
#
# lbl = tk.Label(root, text="Personality Prediction Using Machine Learning", font=('times', 35,' bold '), height=1, width=60,bg="#EE3A8C",fg="white")
# lbl.place(x=0, y=10)


frame_alpr = tk.LabelFrame(root, text=" --Process-- ", fg="white",width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="#191970")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=90)

    
    
###########################################################################
# def train_model():
 
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X= CNNModel.main()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

#     print(msg)

############################################################
def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=70, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=200, y=450)

###############################################################################



    
#############################################################################
    
def openimage():
   
    global fn
    fileName = askopenfilename(initialdir=r'D:\100% personality prediction cnn&svm\data', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])



    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    img.image = imgtk
    img.place(x=300, y=100)
  
#############################################################################    

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)

def SVMModel_test(pth):
    def fd_hu_moments(image):
    #For Shape of signature Image
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def quantify_image(image):
        features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1")
    
        # return the feature vector
        return features

    with open('D:/100% personality prediction cnn&svm/data/clf_SVM.pkl', 'rb') as f:
        clf_SVM = pickle.load(f)
        
    
    

    image = cv2.imread(pth)
    
    # pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
       
    # quantify the image and make predictions based on the extracted
    # features using the last trained Random Forest
    
    features1 = quantify_image(image)
    features2 = fd_hu_moments(image)
    global_feature = np.hstack([features1,features2])
    
    preds =clf_SVM.predict([global_feature])
    
    if preds[0]==2:
        Cd="Conscientiousness"
    # elif preds[0]==0:
    #     Cd="disgusted"
    elif preds[0]==3:
        Cd="Openness"
    # elif preds[0]==1:
    #     Cd="disgusted"  
    elif preds[0]==4:
        Cd="Agreeableness"
    elif preds[0]==5:
        Cd="Neuroticism" 
    elif preds[0]==6:
        Cd="Extraversion"     
    # if preds[0]==1:
    #     label="Signature is Real !!!"
    # else:
    #     label="Signature is Forged !!!"
    
    return Cd



def testSVM_model():
    global fn

    if fn!="":
        update_label("SVM Model Testing Start...............")
        
        start = time.time()
    
        X=SVMModel_test(fn)
        
        X1="Selected  {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.2} seconds \n".format(end-start)
        
        msg="Image SVM Testing Completed.."+'\n'+ X1 + '\n'+ ET
#        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    



#################################################################################################################
def window():
    root.destroy()





button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="#8EE5EE",fg="white")
button1.place(x=10, y=50)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="#7AC5CD",fg="white")
button2.place(x=10, y=100)

button5 = tk.Button(frame_alpr, text="SVM Prediction", command=testSVM_model,width=15, height=1, font=('times', 15, ' bold '),bg="#53868B",fg="white")
button5.place(x=10, y=150)





exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=10, y=200)



root.mainloop()