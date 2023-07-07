

# -*- coding: utf-8 -*-
#from tracemalloc import stop
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests

import spacy

nlp = spacy.load("en_core_web_sm")

import nltk
from nrclex import NRCLex

import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet = WordNetLemmatizer()
import re
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import string
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import collections
from textblob import TextBlob
import sys

# For DL Model
import keras
from keras.preprocessing import text, sequence
from nltk.tokenize import word_tokenize
max_words = 15000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
model_dl = keras.models.load_model('model')


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

with open("negative-words.txt","r", encoding='latin-1') as neg:
    negwords = neg.read().split("\n")

with open("positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")

with open("afinn2.txt","r") as affin:
    affinity = affin.read().split("\n")

affinity_data = pd.read_csv('afinn2.txt', sep="\t", header=None, names=["word", "value"])
affinity_scores = affinity_data.set_index('word')['value'].to_dict()
sentiment_lexicon = affinity_scores

#st.set_page_config(layout="wide")
#st1, st2, st3 = st.columns((15,1,1))


#def local_css(filename):
#  with open(filename) as f:
#    st1.caption('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#local_css("/content/style.css")

#st.title("Hello world")


#st.title('Summary Extraction with Sentiment Analysis')
#st.markdown("<h2 style='text-align: center; color: black;'>Summary Extraction with Sentiment Analysis</h2>", unsafe_allow_html=True)





def preprocess_summary(summary_dataframe):
  filtered_sum=[]
  filtered_sent=[]
  summary = [x.strip() for x in summary_dataframe]

  for i in range(len(summary)):
    summary_ = re.sub("[^A-Za-z" "]+"," ",summary[i])
    summary_ = re.sub("[0-9" "]+"," ",summary[i])
    
    summary_ = summary_.lower()
    summary_ =summary_.split()
    summary_ = [wordnet.lemmatize(word) for word in summary_ if not word in set(stopwords.words('english'))]
    summary_ = ' '.join(summary_)
    filtered_sum.append(summary_)
    text_tf = tf.fit_transform(filtered_sum)
    feature_names = tf.get_feature_names()
    dense = text_tf.todense()
    denselist = dense.tolist()
    summary_df =pd.DataFrame(denselist, columns=feature_names)
    
    return summary_df, filtered_sum


def word_cloud(summary_df):
  cloud = ' '.join(summary_df)

  f, axes = plt.subplots(figsize=(10,5))
  wordcloud= WordCloud(
        background_color = 'white',
        width = 1800,
        height =1400).generate(cloud)
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.tight_layout(pad=100)
  plt.show()


def word_cloud_positive(summary_df):
  f, axes = plt.subplots(figsize=(10,5))
  pos_words = ' '.join([w for w in summary_df if w in poswords])

  cloud_pos = WordCloud(
        background_color = 'white',
        width =1800,
        height=1400).generate(pos_words)
  plt.imshow(cloud_pos)
  plt.axis("off")
  plt.show()


def word_cloud_negative(summary_df):
  f, axes = plt.subplots(figsize=(10,5))
  neg_words = ' '.join([w for w in summary_df if w in negwords])

  cloud_neg = WordCloud(
        background_color='white',
        width =1800,
        height =1400).generate(neg_words)
  plt.imshow(cloud_neg)

  plt.axis("off")
  plt.show()


  # Parts of speech distribution Analysis
def get_pos_tags(sentences, tagset='universal'):
  #Create the Dataframe to store the count of tags
  df = pd.DataFrame(columns=['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X'])
  for sent in sentences:
      # Extract the part of Speech tags in the sentence
      pos_tags = Counter([j for i,j in nltk.pos_tag(word_tokenize(sent), tagset='universal')])
      #Appends the pos tags to the dataframe, fill NaN values with 0
      df = df.append(pos_tags, ignore_index=True).fillna(0)



  fig = plt.figure(figsize =(10, 7))
  data = df[0:1].values[0]
  col = df.columns.values
 
  # Horizontal Bar Plot
  plt.bar(col, data)
 
  # Show Plot
  plt.show()
  
  
def calc_subj(sum_):
  subj = TextBlob(sum_).sentiment.subjectivity
  polar = TextBlob(sum_).sentiment.polarity
  
  return subj, polar

def emotion_score(summary_em):
  
  
  anger=[];disgust=[];fear=[];joy=[];surprise=[];trust=[];anticipation=[];sadness=[];positive=[];negative=[]
  emotions= ["anger","disgust","fear","joy","surprise","trust","anticipation","sadness","positive","negative"]

  
  emotion = NRCLex(summary_em)

  if "positive" in emotion.raw_emotion_scores.keys():
    positive.append(emotion.raw_emotion_scores['positive'])
  else:
    positive.append(0)


  if "anger" in emotion.raw_emotion_scores.keys():
    anger.append(emotion.raw_emotion_scores['anger'])
  else:
    anger.append(0)

  if "disgust" in emotion.raw_emotion_scores.keys():
    disgust.append(emotion.raw_emotion_scores['disgust'])
  else:
    disgust.append(0)

  if "fear" in emotion.raw_emotion_scores.keys():
    fear.append(emotion.raw_emotion_scores['fear'])
  else:
    fear.append(0)

  if "joy" in emotion.raw_emotion_scores.keys():
    joy.append(emotion.raw_emotion_scores['joy'])
  else:
    joy.append(0)

  if "surprise" in emotion.raw_emotion_scores.keys():
    surprise.append(emotion.raw_emotion_scores['surprise'])
  else:
    surprise.append(0)

  if "trust" in emotion.raw_emotion_scores.keys():
    trust.append(emotion.raw_emotion_scores['trust'])
  else:
    trust.append(0)

  if "anticipation" in emotion.raw_emotion_scores.keys():
    anticipation.append(emotion.raw_emotion_scores['anticipation'])
  else:
    anticipation.append(0)

  if "sadness" in emotion.raw_emotion_scores.keys():
    sadness.append(emotion.raw_emotion_scores['sadness'])
  else:
    sadness.append(0)

  if "negative" in emotion.raw_emotion_scores.keys():
    negative.append(emotion.raw_emotion_scores['negative'])
  else:
    negative.append(0)
  
  emotions_df = pd.DataFrame(list(zip(anger, anticipation, disgust, fear, joy, negative,positive, sadness, surprise, trust)),
               columns =['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
       'positive', 'sadness', 'surprise', 'trust'])
  
  fig = plt.figure(figsize =(10, 7))
  data = emotions_df[0:1].values[0]
  col = emotions_df.columns.values
 
  # Horizontal Bar Plot
  plt.barh(col, data)
 
  # Show Plot
  plt.show()
  
 

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score






Category=  pd.read_csv(r"Category.csv")
# select Category of Book like Sport,Art,Science etc.

#url1 = 'https://www.goodreads.com/shelf/show/'+str(choice)
def get_choice(url1):
    
    #selecting Book Title
    try:
        Book_Title = []
        req = requests.get(url1)
        content = bs(req.content,'html.parser')
        book = content.find('div',class_ = 'elementList')
        for each in book:
            spec = each.find_all_next('a',class_ = 'bookTitle')
            for i in spec:
                Book_Title.append(i.text)
        x = Book_Title[:50]
        del Book_Title
        
        return x
        

    except:
        return st.write("Please Check Your Internet Connection")
  

        
def sentiment_wrapper(): 
    choice = st.sidebar.selectbox('Select Category',Category['Category'])       
    Book_Title = get_choice('https://www.goodreads.com/shelf/show/'+str(choice))
    
    
    if Book_Title != None:
        Book = st.sidebar.selectbox('Select Book',Book_Title)
        #get Book Summary:
        Book_urls = []
        req = requests.get('https://www.goodreads.com/shelf/show/'+str(choice))
        content = bs(req.content,'html.parser')
        
        Bookdetails = content.find_all('div', class_ = 'elementList')
        for book in Bookdetails:
            book_anchors = book.find('a')
            Book_url = 'https://www.goodreads.com' + book_anchors.get('href')
            Book_urls.append(Book_url)
        y = np.array(Book_urls[:50])
        del Book_urls
    
    
        for i,j in enumerate(Book_Title):
            if j == str(Book):
                url = y[i]
                #st1.caption(url)
                req = requests.get(url)
                content = bs(req.content,'html.parser') 
                try:
                    summary_=""
                    summary = content.find('div',class_ = 'TruncatedContent')
                    summary_ = summary.text
                    #st.write(summary_[:-8])
                    #st.info(summary_[:-8])
                    #book_summ = process_summary(summary_[:-8])
                    book_data_st = pd.DataFrame({'summary': [summary_[:-8]]}) 
    
    
    
    
    
                    #summary_df = preprocess_summary(book_data_st.summary)
                    
              
                    with st.expander("Book Summary"):
                      st.info(summary_[:-8])
    
                    with st.expander("Book URL"):
                     
                      st.caption(url)
                      #st.table(book_data_st)
    
                    with st.expander("Most Common Words"):
                      summ_df,filtered_summary = preprocess_summary(book_data_st.summary)
                      #st.table(preprocess_summary(summ_df))
                      st.set_option('deprecation.showPyplotGlobalUse', False)
                      st.pyplot(word_cloud(summ_df), figsize =(7, 4))
                      
    
                    with st.expander("Positive Words"):
                      summ_df, filtered_summary= preprocess_summary(book_data_st.summary)
                      #st.table(preprocess_summary(summ_df))
                      st.set_option('deprecation.showPyplotGlobalUse', False)
                      st.pyplot(word_cloud_positive(summ_df),figsize =(7, 4))
    
                    with st.expander("Negative Words"):
                      summ_df,filtered_summary = preprocess_summary(book_data_st.summary)
                      #st.table(preprocess_summary(summ_df))
                      st.set_option('deprecation.showPyplotGlobalUse', False)
                      st.pyplot(word_cloud_negative(summ_df), figsize =(7, 4))
    
                    
                    with st.expander("Parts of Speech Distribution (POS)"):
                      summ_df,filtered_summary = preprocess_summary(book_data_st.summary)
                      st.pyplot(get_pos_tags(filtered_summary), figsize =(7, 4))
    
                    with st.expander("Subjectivity and Polarity"):
                      st.write(calc_subj(book_data_st.summary.values[0]))
    
    
    
                    with st.expander("Emotion scores"):
                      st.pyplot(emotion_score(book_data_st.summary.values[0]), figsize =(7, 4))
    
                    
    
                    with st.expander("Sentiment Value"):
                      summ_df,filtered_summary = preprocess_summary(book_data_st.summary)
                      st.write(calculate_sentiment(book_data_st.summary.values[0]))
    
                   

                except:
                    st.caption('Sorry We are Unable to Find Summary For this Book')    
