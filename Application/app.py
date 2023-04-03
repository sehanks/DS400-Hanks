import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os


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
        st.subheader('Upload Audio File')
        
        with st.container():
            column1, column2 = st.columns(2)
            with column1:
                audio_file = st.file_uploader('Upload audio file', type = ['wav', 'mp3', 'ogg'])  # File uploader
                if audio_file is not None:
                    if not os.path.exists('audio'):  # Check whether the specified path exists or not
                        os.makedirs('audio')  # Create a directory recursively               
                    path = os.path.join('audio', audio.name)  # Join different path components
                    save_audio_file = save_audio_file(audio)  # save_audio_file function
                    if save_audio_file == 1:
                        st.warning('File size is too large. Try another file.')
                    elif save_audio_file == 0:
                        st.audio(audio_file, format = 'audio/wav', start_time = 0)
                        try:
                            wav, sr = librosa.load(path, sr = 45000)
                            X = get_melspec(path)[1]
                            mfcc = librosa.feature.mfcc(wav, sr=sr)
                        except Exception as e:
                            audio_file = None
                            st.error(f'Error {e} - wrong format of the file. Try another .wav file.')
                    else:
                        st.error('Unknown error')
        
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
