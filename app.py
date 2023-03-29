import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2

import IPython
import librosa 
import librosa.display
from IPython.display import Audio # to play the audio files
plt.style.use('seaborn-white')
from python_speech_features import mfcc

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io

from tensorflow.keras.models import load_model
from keras.layers.normalization import layer_normalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Load model
model = load_model('model_cnn.hdf5')

# Datetime
starttime = datetime.now()

# Function to save the audio file
def save_audio_file(file):
    
    if file.size > 5000000:
        return 1
    
    folder = 'audio'
    
    # Convert current date and time to a string
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Clear folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # Check whether the specified path is an existing file or not
            # Check whether the given path represents an existing directory entry that is a symbolic link or not
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Deletes the file path
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open('log0.txt', 'a') as f:
            f.write(f'{file.name} - {file.size} - {date};\n')
    except:
        pass

    with open(os.path.join(folder, file.name), 'wb') as f:
        # Returns a generated PDF document as a byte array
        f.write(file.getbuffer())
        
    return 0

# Creating the app
def main():
    
    # Image
    #side_img = Image.open("___.jpg")
    #with st.sidebar:
        #st.image(side_img, width = 300)
    
    # Set title of app
    st.title('Speech Emotion Recognizer App')
    
    # Dropdown categories for sidebar
    menu = ['Emotion Recognition', 'Project Summary', 'About']
    
    # Title of dropdown
    page = st.sidebar.selectbox('Menu', menu)
    
    # Emotion Recognition page
    if page == 'Emotion Recognition':
        
        # Title of this page
        st.subheader('Emotion Recognition')
        
        # Upload file heading
        st.markdown("##### Upload audio file")
        
        with st.container:
            col1, col2 = st.columns(2)
            with col1:
                # File uploader
                audio = st.file_uploader('Upload your audio file', type = ['wav', 'mp3'])
                if audio is not None:
                    # Check whether the specified path exists or not
                    if not os.path.exists('audio'):
                        # Create a directory recursively
                        os.makedirs('audio')
                    # Join different path components
                    path = os.path.join('audio', audio.name)
                    # save_audio_file function
                    save_audio_file = save_audio_file(audio)
                  
        
        
        #with  st.form(key = 'emotion_clf_form'):
          #raw_text = st.text_area('Type here')
          #submit_text = st.form_submit_button(label = 'Submit')
        #if submit_text:
            #col1, col2 = st.beta_columns(2)
            # Apply functions
            #prediction = predict_emotion(raw_text)
            #probability = get_prediction_prob(raw_text)
            #with col1:
                #st.success('Original Text')
                #st.success('Prediction')
                #st.write(prediction)
            #with col2:
                #st.success('Prediction Probability')
                #st.write(probability)
        
    # Project Summary page
    elif page == 'Project Summary':
        st.subheader('Project Summary')
        
    # About (me) page
    else:
        st.subheader('About')
        
        
if __name__ == '__main__':
    main()
