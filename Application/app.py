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


#model = load_model('model_cnn.hdf5')


starttime = datetime.now()



def predict_emotion(docx):
    results = model.predict([docx])    
    return results



def get_prediction_prob(docx):    
    results = model.predict_prob([docx])    
    return results



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



def waveplot(array, sampling_rate, emotion):
    plt.figure(figsize = (8, 3))
    plt.title('Digital Representation of ' + emotion, size = 15)
    librosa.display.waveplot(array, sr = sampling_rate)
    plt.show()

    

def spectrogram(array, sampling_rate, emotion):
    x = librosa.stft(array)    
    x_db_scale = librosa.amplitude_to_db(abs(x))    
    plt.figure(figsize = (8, 3))    
    plt.title("Spectrogram of " + emotion, size = 15)    
    librosa.display.specshow(x_db_scale, sr = sampling_rate, x_axis = 'time', y_axis = 'hz')
    

    
    

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
                st.markdown('#### Upload Audio File')
                audio_file = st.file_uploader('  ', type = ['wav', 'mp3', 'ogg'])  # File uploader
                if audio_file is not None:
                    if not os.path.exists('audio'):  # Check whether the specified path exists or not
                        os.makedirs('audio')  # Create a directory recursively               
                    path = os.path.join('audio', audio_file.name)  # Join different path components
                    save_audio = save_audio_file(audio_file)  # save_audio_file function
                    if save_audio == 1:
                        st.warning('File size is too large. Try another file.')
                    elif save_audio == 0:
                        st.audio(audio_file, format = 'audio/wav', start_time = 0)
                        try:
                            with column2: 
                                st.markdown('##### Waveplot for Audio File')
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
                        except Exception as e:
                            audio_file = None
                            st.error(f'Error {e} - wrong format of the file. Try another .wav file.')
                    else:
                        st.error('Unknown error')
                else:
                    column3, column4 = st.columns(2)
                    with column3:  
                        if st.button('Try test audio file'): # Test audio file button
                            st.audio(data = 'Application/OAF_back_angry.wav', format = 'audio/wav', start_time = 0) 
                            path = 'Application/OAF_back_angry.wav'
                            with column2: 
                                st.markdown('##### Waveplot for Test Audio File')
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
                                st.markdown('##### Mel-Spectrogram for Test Audio File')
                                fig2 = plt.figure(figsize = (20, 8))
                                librosa.feature.melspectrogram(wav, sr = 45000)
                                S = librosa.feature.melspectrogram(y = wav, sr = 45000, n_mels = 128, fmax = 8000)
                                S_dB = librosa.power_to_db(S, ref = np.max)
                                img = librosa.display.specshow(S_dB, x_axis = 'time', y_axis = 'mel', sr=sr, fmax = 8000)
                                fig2.colorbar(img)
                                st.write(fig2)
                    with column4:
                        if audio_file is None:
                            if st.button('Record an audio file'):  # Record audio button                    
                                with st.spinner(f'Recording for 5 seconds ....'):
                                    st.write('Recording...')
                                    time.sleep(3)
                                    st.success('Recording completed.')
                                    st.write('Error while loading the file.')
                                with column2: 
                                    st.markdown('##### Waveplot for Recorded Audio File')
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
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x = tess['Emotions'], data = tess)
        st.pyplot(fig)
        
    # About (me) page
    else:
        st.title('About')
        
        
if __name__ == '__main__':
    main()
