import streamlit as st
import pandas as pd
# # import sklearn
# import itertools
import numpy as np
# # import seaborn as sb
# import re
# # import nltk
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt
# from sklearn.linear_model import PassiveAggressiveClassifier
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords

st.title('Fake News Detection Results')
train_df = pd.read_csv(r'train.csv')
st.text(train_df.head(15))


st.subheader('File Distribution')
# def create_distribution(dataFile):
#     return sb.countplot(x='Label', data=dataFile, palette='hls')

# st.bar_chart(create_distribution(train_df))

# st.subheader('Data Quality')
# def data_qualityCheck():
#     print("Checking data qualitites...")
#     train_df.isnull().sum()
#     train_df.info()  
#     print("check finished.")
# st.text(data_qualityCheck())

# st.subheader('Data After Index Reset')
# st.text(train_df.shape)
# st.text(train_df.head(10))
# train_df.reset_index(drop= True,inplace=True)
# st.text(train_df.shape)
# st.text(train_df.head(10))

# st.subheader('Label')
# label_train = train_df.Label
# st.text(label_train.head(10))