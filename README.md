# DS400-Hanks
Data Science Capstone Project

### Commit 1: Data Collection and Preprocessing
In this commit I imported all of the necessary libraries, read in the dataset, and completed basic preprocessing. For this project, I am using the Toronto Emotional Speech Set (TESS) from the Northwestern University Auditory Test No. 6. This dataset was particularly appealing because it solely includes females and yet the audio is of such good caliber. Due to other datasets' vast amount of male speakers, there is an imbalance in representation. Two women (26 and 64 years old) recited a set of 200 keywords in the sentence "Say the word _," and recordings evoking each of the following emotions were made (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are a total of 2800 audio files. Each of the two females and their emotions is contained within their own folder in the dataset. The 200 target words audio files (in WAV format) are contained within those.


After collecting the data, I was able to read in the folders with all of the audio files in them, then create a dataframe with each of the file's designated emotion and path. After doing a few basic preprocessing steps I noticed that there was an emotion with no path and after looking into the issue, decided to drop that row.
