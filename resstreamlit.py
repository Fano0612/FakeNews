import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

st.title('Fake News Detection Results')
st.subheader('File Readings')
train_df = pd.read_csv(r'train.csv')
st.text(train_df.head(15))


st.subheader('File Distribution')
def create_distribution(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='hls')

st.bar_chart(create_distribution(train_df))

st.subheader('Data Quality')
def data_qualityCheck():
    print("Checking data qualitites...")
    train_df.isnull().sum()
    train_df.info()  
    print("check finished.")
st.text(data_qualityCheck())

st.subheader('Data After Index Reset')
st.text(train_df.shape)
st.text(train_df.head(10))
train_df.reset_index(drop= True,inplace=True)
st.text(train_df.shape)
st.text(train_df.head(10))

st.subheader('Label')
label_train = train_df.Label
st.text(label_train.head(10))

st.subheader('Drop Label')
train_df = train_df.drop("Label", axis = 1)
st.text(train_df.head(10))
train_df['Statement'][2188]

st.subheader('Stopwords')
lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))
st.text(stpwrds)

st.subheader('Training')
X_train, X_test, Y_train, Y_test = train_test_split(train_df['Statement'], label_train, test_size=0.3, random_state=1)
st.text(X_train)
st.text(Y_train)

st.subheader('TfidfVectorizer')
tfidf_v = TfidfVectorizer()
tfidf_X_train = tfidf_v.fit_transform(X_train)
tfidf_X_test = tfidf_v.transform(X_test)
st.text(tfidf_X_train.shape)

st.subheader('Confussion Matrix Plotting')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classifier = PassiveAggressiveClassifier()
classifier.fit(tfidf_X_train,Y_train)

Y_pred = classifier.predict(tfidf_X_test)
score = metrics.accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {round(score*100,2)}%')
cm = metrics.confusion_matrix(Y_test, Y_pred)
st.bar_chart(plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data']))
st.text(pickle.dump(classifier,open('./model.pkl', 'wb')))
st.text(loaded_model = pickle.load(open('./model.pkl', 'rb')))
