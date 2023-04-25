import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import librosa 
import librosa.display
from tensorflow.keras.models import load_model
import tensorflow as tf
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from io import BytesIO
from pydub import AudioSegment, effects
import ffmpeg


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']


model = load_model('Application/model_cnn.hdf5')


starttime = datetime.now()


def save_audio_file(file):    
    if file.size > 4000000:
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

  

def spectrogram(array, sampling_rate):
    x = librosa.stft(array)
    x_db_scale = librosa.amplitude_to_db(abs(x))   
    librosa.display.specshow(x_db_scale, sr = sampling_rate, x_axis = 'time', y_axis = 'hz')
  


def noise(array):    
    noise_aug = np.random.uniform() * np.amax(array) * 0.06     
    array = (noise_aug * np.random.normal(size = array.shape[0])) + array     
    return array
def pitch(array, sampling_rate):    
    return librosa.effects.pitch_shift(y = array, sr = sampling_rate, n_steps = 1.1)
def slow(array, sampling_rate = 0.4):    
    return librosa.effects.time_stretch(y = array, rate = sampling_rate)
def fast(array, sampling_rate = 1.5):    
    return librosa.effects.time_stretch(y = array, rate = sampling_rate)
def shift(array):    
    shift_aug = int(np.random.uniform(low = -20, high = 10) * 1000)    
    return np.roll(a = array, shift = shift_aug)



def extract_feats(array, sampling_rate):  
    # MFCC
    result = np.array([])
    mfcc = librosa.feature.mfcc(y = array, sr = sampling_rate)
    mfcc_mean = np.mean(mfcc.T, axis = 0)
    result = np.hstack((result, mfcc_mean))  # Horizontal Stack    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y = array)
    zcr_mean = np.mean(zcr.T, axis = 0)
    result = np.hstack((result, zcr_mean))  # Horizontal Stack    
    # Mel Spectogram
    melspec = librosa.feature.melspectrogram(y = array, sr = sampling_rate)
    melspec_mean = np.mean(melspec.T, axis = 0)
    result = np.hstack((result, melspec_mean))  # Horizontal Stack
    # Root Mean Square Value
    rmsv = librosa.feature.rms(y = array)
    rmsv_mean = np.mean(rmsv.T, axis = 0)
    result = np.hstack((result, rmsv_mean))  # Horizontal Stack    
    # Chroma
    stft = np.abs(librosa.stft(array))
    chroma_stft = librosa.feature.chroma_stft(S = stft, sr = sampling_rate)
    chroma_stft_mean = np.mean(chroma_stft.T, axis = 0)
    result = np.hstack((result, chroma_stft_mean))  # Horizontal Stack      
    # Spectral Centroid
    spectral = librosa.feature.spectral_centroid(y = array)
    spectral_mean = np.mean(spectral.T, axis = 0)
    result = np.hstack((result, spectral_mean))  # Horizontal Stack     
    return result


    
def get_feats(path):    
    # Duration and offset takes care of the noise, pitch, slow down, etc.
    array, sampling_rate = librosa.load(path, duration = 3, offset = 0.6)    
    # Normal Audio
    resample_norm = extract_feats(array, sampling_rate)
    result = np.array(resample_norm)    
    # Noise
    get_noise = noise(array)
    resample_noise = extract_feats(get_noise, sampling_rate)
    result = np.vstack((result, resample_noise))  # Vertical Stack    
    # Pitch
    get_pitch = pitch(array, sampling_rate)
    resample_pitch = extract_feats(get_pitch, sampling_rate)
    result = np.vstack((result, resample_pitch))  # Vertical Stack    
    # Slow Down
    get_slow = slow(array)
    resample_slow = extract_feats(get_slow, sampling_rate)
    result = np.vstack((result, resample_slow))  # Vertical Stack    
    # Speed Up
    get_fast = fast(array)
    resample_fast = extract_feats(get_fast, sampling_rate)
    result = np.vstack((result, resample_fast))  # Vertical Stack    
    # Shift
    get_shift = shift(array)
    resample_shift = extract_feats(get_shift, sampling_rate)
    result = np.vstack((result, resample_shift))  # Vertical Stack    
    return result



def get_feats_recorded(wav, sampling_rate):     
    # Normal Audio
    resample_norm = extract_feats(wav, sampling_rate)
    result = np.array(resample_norm)    
    # Noise
    get_noise = noise(wav)
    resample_noise = extract_feats(get_noise, sampling_rate)
    result = np.vstack((result, resample_noise))  # Vertical Stack    
    # Pitch
    get_pitch = pitch(wav, sampling_rate)
    resample_pitch = extract_feats(get_pitch, sampling_rate)
    result = np.vstack((result, resample_pitch))  # Vertical Stack    
    # Slow Down
    get_slow = slow(wav)
    resample_slow = extract_feats(get_slow, sampling_rate)
    result = np.vstack((result, resample_slow))  # Vertical Stack    
    # Speed Up
    get_fast = fast(wav)
    resample_fast = extract_feats(get_fast, sampling_rate)
    result = np.vstack((result, resample_fast))  # Vertical Stack    
    # Shift
    get_shift = shift(wav)
    resample_shift = extract_feats(get_shift, sampling_rate)
    result = np.vstack((result, resample_shift))  # Vertical Stack    
    return result



def get_pred(path):
    onehot = OneHotEncoder() 
    np_onehot = np.array(emotions).reshape(-1, 1)
    y = onehot.fit_transform(np_onehot).toarray()
    feat = get_feats(path)
    sc = StandardScaler()
    feat_fit = sc.fit_transform(feat)
    expand_dim = np.expand_dims(feat_fit, axis = 2)
    pred = model.predict(expand_dim)
    y_pred = onehot.inverse_transform(pred)
    return y_pred.flatten()



def get_pred_recorded(path, sr):
    onehot = OneHotEncoder() 
    np_onehot = np.array(emotions).reshape(-1, 1)
    y = onehot.fit_transform(np_onehot).toarray()
    feat = get_feats_recorded(path, 45000)
    sc = StandardScaler()
    feat_fit = sc.fit_transform(feat)
    expand_dim = np.expand_dims(feat_fit, axis = 2)
    pred = model.predict(expand_dim)
    y_pred = onehot.inverse_transform(pred)
    return y_pred.flatten()



def main():
    
    # Image
    sidebar_image = Image.open('Application/emotion2.png')
    with st.sidebar:
        st.image(sidebar_image, width = 300)
    
    # Dropdown categories for sidebar
    menu = ['Emotion Recognition', 'Project Summary', 'About']
    
    # Title of dropdown
    page = st.sidebar.selectbox('Menu', menu)
    
    # Emotion Recognition page
    if page == 'Emotion Recognition':
        st.title('Speech Emotion Recognizer App')
        
        with st.container():
            column1, column2 = st.columns(2)
            with column1:
                audio_file = st.file_uploader('Upload Audio File', type = ['wav', 'mp3', 'ogg'])  # File uploader
                if audio_file is not None:
                    if not os.path.exists('audio'):  # Check whether the specified path exists or not
                        os.makedirs('audio')  # Create a directory recursively               
                    path = os.path.join('audio', audio_file.name)  # Join different path components
                    save_audio = save_audio_file(audio_file)  # save_audio_file function
                    if save_audio == 1:
                        st.warning('File size is too large. Try another file.')
                    elif save_audio == 0: 
                        with column2:
                            st.markdown('#  ')
                            st.markdown('#  ')
                            st.audio(audio_file, format = 'audio/wav', start_time = 0)
                        try:
                            with column1: 
                                st.markdown('#  ')
                                st.markdown('###### Waveplot for Audio File')
                                fig = plt.figure(figsize = (20, 8))
                                wav, sr = librosa.load(path, sr = 45000)
                                librosa.display.waveplot(wav, sr = 45000)
                                plt.gca().axes.get_yaxis().set_visible(False)
                                plt.gca().axes.get_xaxis().set_visible(False)
                                plt.gca().axes.spines['right'].set_visible(False)
                                plt.gca().axes.spines['left'].set_visible(False)
                                plt.gca().axes.spines['top'].set_visible(False)
                                plt.gca().axes.spines['bottom'].set_visible(False)
                                st.write(fig)
                            with column2:
                                st.markdown('#  ')
                                st.markdown('#  ')
                                st.markdown('######  ')
                                st.markdown('###### Mel-Spectrogram for Audio File')
                                fig2 = plt.figure(figsize = (20, 8))
                                spectrogram(wav, sr)
                                st.write(fig2)
                        except Exception as e:
                            audio_file = None
                            st.error(f'Error {e} - wrong format of the file. Try another .wav file.')
                    else:
                        st.error('Unknown error')
                    
                    
                else:
                    column3, column4 = st.columns(2)
                    with column3:
                        if st.button('Try test audio file'): # Test audio file button
                                with column2:
                                    st.markdown('#  ')
                                    st.markdown('#  ')
                                    st.audio(data = 'Application/OAF_back_sad.wav', format = 'audio/wav', start_time = 0) 
                                    path = 'Application/OAF_back_sad.wav'
                                    audio_file = 'test_file'
                                with column1: 
                                    st.markdown('#  ')
                                    st.markdown('###### Waveplot for Test Audio File')
                                    fig = plt.figure(figsize = (20, 8))
                                    wav, sr = librosa.load(path, sr = 45000)
                                    librosa.display.waveplot(wav, sr = 45000)
                                    plt.gca().axes.get_yaxis().set_visible(False)
                                    plt.gca().axes.get_xaxis().set_visible(False)
                                    plt.gca().axes.spines['right'].set_visible(False)
                                    plt.gca().axes.spines['left'].set_visible(False)
                                    plt.gca().axes.spines['top'].set_visible(False)
                                    plt.gca().axes.spines['bottom'].set_visible(False)
                                    st.write(fig)
                                with column2:
                                    st.markdown('#  ')
                                    st.markdown('####  ')
                                    st.markdown('######   ')
                                    st.markdown('###### Mel-Spectrogram for Test Audio File')
                                    fig2 = plt.figure(figsize = (20, 8))
                                    spectrogram(wav, sr)
                                    st.write(fig2)
                        else: 
                            with column4:
                                audio = audiorecorder('Click to record', 'Recording...')
                            with column2:
                                st.markdown('#  ')
                                st.markdown('#  ')
                                if len(audio) > 0:
                                    st.audio(audio.tobytes()) 
                            with column1:
                                st.markdown('###  ')
                                st.markdown('###### Waveplot for Recorded Audio File')
                                fig = plt.figure(figsize = (20, 8))
                                librosa.display.waveplot(audio.astype(np.float32), sr = 45000)
                                plt.gca().axes.get_yaxis().set_visible(False)
                                plt.gca().axes.get_xaxis().set_visible(False)
                                plt.gca().axes.spines['right'].set_visible(False)
                                plt.gca().axes.spines['left'].set_visible(False)
                                plt.gca().axes.spines['top'].set_visible(False)
                                plt.gca().axes.spines['bottom'].set_visible(False)
                                st.write(fig)
                            with column2:
                                st.markdown('#  ')
                                st.markdown('####  ')
                                st.markdown('######   ')
                                st.markdown('###### Mel-Spectrogram for Recorded Audio File')
                                fig2 = plt.figure(figsize = (20, 8))
                                spectrogram(audio.astype(np.float32), 45000)
                                st.write(fig2)
                        with st.container():
                            #with column1: 
                            pred_emotion = get_pred_recorded(audio.astype(np.float32), 45000)
                            unique, counts = np.unique(pred_emotion, return_counts = True)
                            st.markdown('## Emotion Detected: {}'.format(unique[counts.argmax()]))
                        with st.container():
                            with column2:
                                fig = plt.figure(figsize = (15, 7))
                                unique, counts = np.unique(pred_emotion, return_counts = True)
                                bar = np.asarray((unique, counts)).T
                                col = list(bar[:, 0])
                                row = list(bar[:, 1])
                                color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                                plt.bar(x = col, height = [int(x) for x in row], color = color)
                                plt.xlabel('Emotions Detected')
                                plt.ylabel('Frequency')
                                plt.title('Frequency of Emotions Detected')
                                st.pyplot(fig)
                                
    
        if audio_file is not None:
            if not audio_file == 'test_file':
                st.markdown('#  ')
                st.markdown('###### Analysis of Audio File')  # Show details of the audio file in the menu bar
                file_details = {'Name': audio_file.name, 'Size': audio_file.size}
                st.write(file_details)
                
            with st.container():
                tess = pd.read_csv('Application/Tess_df.csv')
                
                # Prediction
                if audio_file == 'test_file':
                    pred_emotion = get_pred('Application/OAF_back_sad.wav')
                    unique, counts = np.unique(pred_emotion, return_counts = True)
                    st.markdown('## Emotion Detected: {}'.format(unique[counts.argmax()]))
                if not audio_file == 'test_file':
                    pred_emotion = get_pred(audio_file)
                    unique, counts = np.unique(pred_emotion, return_counts = True)
                    st.markdown('## Emotion Detected: {}'.format(unique[counts.argmax()]))
            with st.container():
                if audio_file == 'test_file':
                    fig = plt.figure(figsize = (15, 7))
                    unique, counts = np.unique(pred_emotion, return_counts = True)
                    bar = np.asarray((unique, counts)).T
                    col = list(bar[:, 0])
                    row = list(bar[:, 1])
                    color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                    plt.bar(x = col, height = [int(x) for x in row], color = color)
                    plt.xlabel('Emotions Detected')
                    plt.ylabel('Frequency')
                    plt.title('Frequency of Emotions Detected')
                    st.pyplot(fig)
                if not audio_file == 'test_file':
                    fig = plt.figure(figsize = (10, 4))
                    unique, counts = np.unique(pred_emotion ,return_counts = True)
                    bar = np.asarray((unique, counts)).T
                    col = list(bar[:, 0])
                    row = list(bar[:, 1])
                    color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                    plt.bar(x = col, height = [int(x) for x in row], color = color)
                    plt.xlabel('Emotions Detected')
                    plt.ylabel('Frequency')
                    plt.title('Frequency of Emotions Detected')
                    st.pyplot(fig)
                        
                
                 
    
    # Project Summary page
    elif page == 'Project Summary':
        st.title('Project Summary')
        st.subheader('Github Link')
        link = '[GitHub repository]' \
               '(https://github.com/sehanks/DS400-Hanks)'
        st.markdown(link, unsafe_allow_html = True)
        
        st.subheader('Dataset Information')
        text = """
            This application is a part of a final Data Science Capstone project. 
            The dataset used in this project is the Toronto Emotional Speech Set (TESS) from the Northwestern University Auditory Test No. 6. 
            This dataset was particularly appealing because it solely includes females and yet the audio is of such good caliber. 
            Due to other datasets' vast amount of male speakers, there is an imbalance in representation. 
            Two women (26 and 64 years old) recited a set of 200 keywords in the sentence "Say the word _," 
            and recordings evoking each of the following emotions were made (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). 
            There are a total of 2800 audio files. Each of the two females and their emotions is contained within their own folder in the dataset. 
            The 200 target words audio files (in WAV format) are contained within those.   
            """
        st.markdown(text, unsafe_allow_html = True)
        
        tess = pd.read_csv('Application/Tess_df.csv')
        fig = plt.figure(figsize = (10, 4))
        sns.countplot(x = tess['Emotions'], data = tess)
        plt.title('Counts of Each Emotion')
        st.pyplot(fig)
        
    # About (me) page
    else:
        st.title('About')
        column1, column2 = st.columns(2)
        with column1:
            st.markdown('### Email')
            st.info('sehanks01@gmail.com')
        with column2:
            linkedin = Image.open('Application/linkedin.png')
            st.image(linkedin, width = 100)
            st.markdown('##  ')
            link = '[Sarah Hanks LinkedIn]' \
               '(www.linkedin.com/in/hanks-sarah)'
            st.markdown(link, unsafe_allow_html = True)
            
        
        
if __name__ == '__main__':
    main()
