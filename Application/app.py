import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


model = load_model('model_cnn.hdf5')


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
        st.markdown("##### Upload audio file"
        
        
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
