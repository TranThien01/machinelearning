from secrets import choice
import streamlit as st
import pandas as pd
import numpy as np
import time
# Importing the Keras libraries and other packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from tensorflow.keras.utils import plot_model
# from IPython.display import Image
from PIL import Image
from tensorflow.keras.models import load_model
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#load model
classifier=load_model('Deeplearning.h5')
import os




st.title("MACHINE LEARNING")
st.write("Object Detection & Classification")

menu=["Cnn","Transfer Learning","Objection Detection"]
choice=st.sidebar.selectbox('Menu',menu)

if choice=='Cnn':

# nhận file upload lên
 file=st.file_uploader('Chon file ảnh',type=['jpg'])

 if not (file is None):
    # Processing image
    if os.path.exists('uploads/') == False:
      os.mkdir('uploads')
    with open(os.path.join("uploads/", file.name), "wb") as f: 
      f.write(file.getbuffer())
    img = Image.open(file)
    st.image(img)
    # fu=file.getvalue()
    test_image=image.load_img("uploads/" + file.name,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    print(test_image.shape)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    print(test_image.shape)

    result=classifier.predict(test_image)
    print(result)
    if result[0][0]>=0.5:
       prediction='TuiThanThienMoiTruong'
    else:
      prediction='TuiNyLon'
    st.title("Phân loại: ")
    st.write(prediction)


    plt.imshow(test_image[0])


    #%pylab inline
    img=mpimg.imread("uploads/" + file.name)
    imgplot = plt.imshow(img)
    plt.show()
elif choice=='Transfer Learning':
   file1=st.file_uploader('Chon file ảnh',type=['jpg'])
   if not (file1 is None):
    # Processing image
    if os.path.exists('uploads/') == False:
      os.mkdir('uploads')
    with open(os.path.join("uploads/", file1.name), "wb") as f: 
      f.write(file1.getbuffer())
    img = Image.open(file1)
    st.image(img)

    classifier=load_model('TuiThanThienMoiTruog_TuiNyLon_Small_vgg16.h5')

    test_image=image.load_img("uploads/" + file1.name,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    print(test_image.shape)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    print(test_image.shape)

    result=classifier.predict(test_image)
    print(result)
    if result[0][0]<result[0][1]:
      prediction='TuiThanThienMoiTruong'
    else:
     prediction='TuiNyLon'
    st.write(prediction)

  
    img=mpimg.imread(file1)
    imgplot = plt.imshow(img)
    plt.show()


#st.video("https://www.youtube.com/watch?v=niuZ3pdoc4I")