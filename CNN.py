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
import pandas as pd
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from tensorflow.keras.models import load_model
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#load model
classifier=load_model('Deeplearning.h5')

# nhận file upload lên

file=st.file_uploader('Chon file ảnh',type=['jpg'])

if not (file is None):
    test_image=image.load_img(file,target_size=(64,64))
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
    print(prediction)


    plt.imshow(test_image[0])


    #%pylab inline
    img=mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.show()


st.video("https://www.youtube.com/watch?v=niuZ3pdoc4I")