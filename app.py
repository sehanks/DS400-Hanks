import streamlit as st
import pandas as pd
import numpy as np


def main():
    
    # Set title of app
    st.title('Speech Emotion Recognizer App')
    # Dropdown categories for sidebar
    menu = ['Emotion Recognition', 'Project Summary', 'About']
    # Title of dropdown
    choice = st.sidebar.selectbox('Menu', menu)
    
    # Emotion Recognition page
    if choice == 'Emotion Recognition':
        # Title of this page
        st.subheader('Emotion Recognition')
        with  st.form(key = 'emotion_clf_form'):
          raw_text = st.text_area('Type here')
          submit_text = st.form_submit_button(label = 'Submit')
        
    # Project Summary page
    elif choice == 'Project Summary':
        st.subheader('Project Summary')
        
    # About (me) page
    else:
        st.subheader('About')
        
        
if __name__ == '__main__':
    main()
