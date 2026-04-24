RythMind: Music Clustering using CVAE, Autoencoder and Spectral Features

Overview

This project focuses on clustering music tracks using unsupervised learning methods. The main idea is to learn useful representations from audio data and optionally combine them with lyrics. Models such as Conditional Variational Autoencoder and Autoencoder are used, and their performance is compared with simpler methods like PCA and direct clustering on audio features.

Dataset

For audio data, the GTZAN Genre and Rabindra Sangeet Dataset is used. It contains 1100 music samples across 11 different genres such as blues, classical, jazz, rock, and others. You can download it from the following link: 
https://marsyas.info/downloads/datasets.html
https://www.kaggle.com/datasets/fsoroni/rabindra-sangeet-audio-and-lyrics

For lyrics, a text dataset containing Drake song lyrics is used. It is a simple text file where each line represents a lyric. You can find similar datasets on Kaggle by searching for Drake lyrics: https://www.kaggle.com/datasets

Dataset Setup

After downloading the datasets, the folder structure should be organized like this:

project folder inside data folder inside audio folder you will have genre folders like blues, jazz, rock and so on inside each genre folder there will be audio files with extension au inside lyrics folder you will keep a file named drake_lyrics.txt

Make sure the audio files are arranged by genre and the lyrics file is a plain text file.

How to Run

First install all required libraries using the requirements file. Then open the notebook file inside the notebooks folder. Run the cells step by step.

The notebook includes everything from feature extraction to model training, clustering, evaluation, and visualization.

Methods Used

The project compares multiple approaches including CVAE using both audio and lyrics, CVAE using only audio, Autoencoder with KMeans clustering, PCA with KMeans, and direct clustering on extracted audio features.

Evaluation

Clustering performance is measured using Silhouette Score, Normalized Mutual Information, Adjusted Rand Index, and Cluster Purity.

Results

The results show that Autoencoder performs better overall by balancing cluster separation and alignment with genre labels. Multi modal learning using lyrics did not improve performance because the lyrics dataset does not match the genre diversity of the audio dataset.

Outputs

All results are saved in the results folder. This includes clustering metrics, comparison results, cluster assignments, and reconstruction examples.

Author

Faria Soroni

CSE 715 Neural Networks Project
