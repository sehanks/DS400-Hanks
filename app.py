import streamlit as st
import pandas as pd
import numpy as np


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
