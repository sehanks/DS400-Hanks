# DS400-Hanks
Data Science Capstone Project

### Dataset: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF 

### App Link: https://sehanks-ds400-hanks-applicationapp-qe8am2.streamlit.app/

### Commit 1: Data Collection and Preprocessing
In this commit I imported all of the necessary libraries, read in the dataset, and completed basic preprocessing. For this project, I am using the Toronto Emotional Speech Set (TESS) from the Northwestern University Auditory Test No. 6. This dataset was particularly appealing because it solely includes females and yet the audio is of such good caliber. Due to other datasets' vast amount of male speakers, there is an imbalance in representation. Two women (26 and 64 years old) recited a set of 200 keywords in the sentence "Say the word _," and recordings evoking each of the following emotions were made (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are a total of 2800 audio files. Each of the two females and their emotions is contained within their own folder in the dataset. The 200 target words audio files (in WAV format) are contained within those.


After collecting the data, I was able to read in the folders with all of the audio files in them, then create a dataframe with each of the file's designated emotion and path. After doing a few basic preprocessing steps I noticed that there was an emotion with no path and after looking into the issue, decided to drop that row.


### Commit 2: Exploratory Data Analysis, Data Augmentation, and Feature Extraction
In this commit, I completed Exploratory Data Analysis (EDA), Data Augmentation, and Feature Extraction. For EDA, I created four different types of visualizations. The first visualization was a count plot of each of the different emotions. This plot once again ensured that there is an equal number of emotions. The next visualization I created was a waveplot, which shows how loud the audio file is at different times. I then created a spectrogram, which is a graphic of the frequency spectrum of sound or other signals as it varies over time. The last visualization I created was a Mel-Frequency Cepstral Coeffecients (MFCC) graph. This graph is specifically used in sound processing and is a representation of the short-term power spectrum of a sound. For each of the seven emotions, I printed out a waveplot, spectrogram, and MFCC graph in order to compare what the different emotions looked like when the same sentence was spoken. In order to get an even better understanding of the emotions, I printed the MFCC graphs side-by-side. This allowed me to see a closer comparison of each emotion in a graph. 


Data Augmentation is the technique of creating copies of a dataset and artifically altering that dataset. By adding minor changes to my dataset, I can generate new data samples with this technique. I can use alterations such as noise injection, time shifting, pitch and speed changes, etc. to produce data for the audio. Making my models resistant to these disturbances will increase their transferability into real-world situations. The label from the training set must be preserved when adding the disturbances for this to work. First I needed to determine which augmentation strategies fit my dataset the best. After attempting many different alterations, I decided to use noise injection, pitch changes, speed changes, and time shifting. 


A crucial step in studying and discovering relationships between various variables is feature extraction. I need to transform the audio data presented into a type that the models can interpret. Since the models cannot simply interpret the information supplied by the audio files, I must turn it into a structure that can be comprehended by them via feature extraction. I can conduct a number of transformations on the data and sampling rate to derive useful information from them. First I created a function in order to extract the features. I decided to use MFCC, Zero Crossing Rate, Mel Spectrogram, Root Mean Squared Value, Chroma Stft, and Spectral Centroid. MFCC forms a cepstral representation, Zero Crossing Rate computes the zero-crossing rate of an audio time series, Mel Spectrogram computes a mel-scaled spectrogram, Root Mean Squared Value computes root-mean-square (RMS) value for each frame, Chroma Stft computes a chromagram from a waveform or power spectrogram, and Spectral Centroid computes the spectral centroid. I then created a function to get those extracted features and perform the data augmentation transformations on them. I will have then applied data augmentation and extracted the features for each audio files and saved them.


### Commit 3: Data Preparation and Splitting Data
After extracting the features, I created a dataframe and saved it into a .csv file. I then created a new python notebook and read in the csv file in order to organize the code. For Data Preparation, I split the data into the data (X) and data labels (y). This is a multiclass classification problem so I must onehot encode the labels (y). Next, I split the data into training and testing set 80/20 then standardized both sets using Standard Scaler. To make data appropriate for modeling, I added a new axis that will be visible at the axis point in the extended array shape.


### Commit 4: Model Implementation and Training
The first model I used was CNN. A deep learning network architecture that learns directly from data is a convolutional neural network (CNN). CNNs are very helpful for recognizing objects, classes, and categories in photos by looking for patterns in the images. They can also be quite useful for categorizing signals, time-series, and audio data. This model was able to achieve about 99% accuracy using 10 epochs and a batch size of 128. I was then able to create a new dataframe of the Predicted Labels vs. the Actual Labels and plot this information into a confusion matrix in order to look at which labels were being misidentified the most. This visualization showed the CNN model was relatively equal in accuracy across all seven emotions. Overall, this was a very successful model.

The second model I used was LSTM. Long short-term memory networks, or LSTMs, are employed in deep learning. Many recurrent neural networks (RNNs) are able to learn long-term dependencies, particularly in tasks involving sequence prediction. This unique version of RNN exhibits exceptional performance on a wide range of issues. With the exception of single data points like photos, LSTM can handle the full sequence of data. This has uses in machine translation and speech recognition, among others. This model was able to achieve about 99% accuracy using 10 epochs. I was then able to create a new dataframe of the Predicted Labels vs. the Actual Labels and plot this information into a confusion matrix in order to look at which labels were being misidentified the most, just like I did for the CNN model. This visualization showed the LSTM model was relatively equal in accuracy across all seven emotions. Overall, this was also a very successful model.


### Commit 5: Parameter Tuning
In this stage, I was able to alter my models to make them each as accurate as possible before choosing my final model. After much consideration, I have chosen CNN to be my final model due to its high accuracy and efficiency. I was also able to create a decoder that takes in an input audio, and prints out the transcript for that audio. This may be useful during my application stage.


### Commit 6: Application Design
In this stage, I was able to begin my application. I have chosen to use StreamLit to build and share my data application. In my app, there will be 3 pages the user can choose from. The first page will be where the Emotion Recognition application is done, the second page will be a Project Summary, and the third page will be about myself as the creator. In the Emotion Recognition page, the user should be able to record an audio file and use my final model in order to predict which emotion is being presented. I will also create visuals in order show the user what emotions may be underlying in their audio. 


## Commit 7: Code Revision
In this stage, I finished up the coding for my application. My project was successful in taking a user's input speech and accurately predicting the audio's emotion.


## Commit 8: Code cleanup, Packaging, and Publishing
In this stage, I cleaned up the my code for each of my files as well as made sure the application was running as intended.
