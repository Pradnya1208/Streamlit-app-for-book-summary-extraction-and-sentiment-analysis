
# -*- coding: utf-8 -*-
#from tracemalloc import stop
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests

#st.title('Page2')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""    
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
from io import BytesIO
from PIL import Image


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


 


def emotion_score(summary_em):
  
  
  anger=[];disgust=[];fear=[];joy=[];surprise=[];trust=[];anticipation=[];sadness=[];
  emotions= ["anger","disgust","fear","joy","surprise","trust","anticipation","sadness"]

  
  emotion = NRCLex(summary_em)



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


  
  emotions_df = pd.DataFrame(list(zip(anger, anticipation, disgust, fear, joy, sadness, surprise, trust)),
               columns =['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'])
  
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



def find_sentiment(summary_dataframe, summ_text):
    
 
  filtered_summary=[]
  summary = [x.strip() for x in summary_dataframe]

  for i in range(len(summary)):
    summary_ = re.sub("[^A-Za-z" "]+"," ",summary[i])
    summary_ = re.sub("[0-9" "]+"," ",summary[i])
    
    summary_ = summary_.lower()
    filtered_summary.append(summary_)
    
  summary_df =pd.DataFrame()
  summary_df["summary"] = filtered_summary
  
  input_summary = summary_df[0:1].summary.values

  # Tokenize the input
  tokenize.fit_on_texts(input_summary)
  tokenized_text = tokenize.texts_to_matrix(input_summary)

  scores = model_dl.predict(tokenized_text, verbose=1, batch_size=32)
  num_ = np.where(scores[0] == max(scores[0]))[0][0]
  if num_ == 0:
    y_pred= "Negative"
  elif num_==1:
    y_pred= "Neutral"
  else:
    y_pred= "Positive"
    
  
    
  
  emotion_pn = NRCLex(summ_text)
  pos_ = emotion_pn.raw_emotion_scores['positive']
  neg_ = emotion_pn.raw_emotion_scores['negative']
  if pos_ > neg_:
      if ((pos_-neg_)/(pos_+neg_))*100 >5:
          y_pred2 = "Positive"
      else:
          y_pred2="Neutral"
          
  elif pos_ < neg_:
      if ((neg_-pos_)/(pos_+neg_))*100 >5:
          y_pred2 = "Negative"
      else:
          y_pred2="Neutral"
  else:
      y_pred2="Neutral"
     
        
  sent_score = 0
  if summ_text:
      sentence = nlp(summ_text)
      for word in sentence:
          sent_score += sentiment_lexicon.get(word.lemma_, 0)
          
      if sent_score > 0:
          y_pred3 = "Positive"
      elif sent_score < 0:
          y_pred3="Negative"
      else:
          y_pred3="Neutral" 
          
          
  # Final results
  if y_pred == y_pred2 == y_pred3:
      result = y_pred2
  else:
      result = y_pred2
      
      
  col1, col2, col3 = st.columns([2, 3, 2])




  with col2:
    if result=="Positive":
        image = Image.open('positive.png')
        st.image(image, caption="Go ahead, it's a great page turner. Happy reading!!", use_column_width=True)
        
    elif result=="Negative":
        image = Image.open('negative.png')
        st.image(image, caption='I would recommend to look for detailed analysis of this book.', use_column_width=True)
        
    else:
        image = Image.open('Neutral.png')
        st.image(image, caption='Looks like this book balances both the extremities very well!! Go ahead, and dig deeper with the help of detailed analysis.', use_column_width=True)



def pie_chart(summ_text):
    
    
  positive=[];negative=[];anger=[];disgust=[];fear=[];joy=[];surprise=[];trust=[];anticipation=[];sadness=[];


  #emotions= ["positive","negative"]
  labels_ = ['Negative', 'Positive']
  colors = ['#D08A78', '#B9CE8B']
  color_pallette = ['#6d9891', '#afac9b', '#e1c1a5', '#f69f82', '#dd826f', '#76575d', '#648C49']
  emotion = NRCLex(summ_text)
  if "negative" in emotion.raw_emotion_scores.keys():
    negative.append(emotion.raw_emotion_scores['negative'])
  else:
    negative.append(0)
    
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


    
  emotions_pn = pd.DataFrame(list(zip(negative,positive)),
               columns =['negative','positive'])
  
  emotions_df = pd.DataFrame(list(zip(anger, anticipation, disgust, fear, joy, sadness, surprise, trust)),
               columns =['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'])

  
  emotions_df = emotions_df.drop(emotions_df.columns[emotions_df.eq(0).any()],1)
  emotions = emotions_df.columns.values
  
  explode = (0.05, 0.05)
  explode2 = ((0.05,)*emotions_df.shape[1])
  width = st.sidebar.slider("plot width", 1, 25, 10)
  height = st.sidebar.slider("plot height", 1, 25, 5)

  fig, ax = plt.subplots(1,2,figsize=(width, height))
  fig.subplots_adjust(hspace=3.0)
  
  
  plt.tight_layout() 
  
  
  # Pie Chart
  ax[0].pie(emotions_pn[0:1].values[0], colors=colors, labels=labels_,
         autopct='%1.1f%%', pctdistance=0.5,
         explode=explode, textprops={'fontsize': 12})
  ax[0].text(0.5, 0.5, 'Sentiment Analysis', transform = ax[0].transAxes, va = 'center', ha = 'center', backgroundcolor = 'white')
  
  
  ax[1].pie(emotions_df[0:1].values[0], colors = color_pallette, labels=emotions,
         autopct='%1.1f%%', pctdistance=0.85,
         explode=explode2, textprops={'fontsize': 10})
  ax[1].text(0.5, 0.5, 'Raw Emotions Analysis', transform = ax[1].transAxes, va = 'center', ha = 'center', backgroundcolor = 'white')

  #ax[0].title('Sentiment Analysis')
  # draw circle
  centre_circle = plt.Circle((0, 0), 0.70, fc='white')
  fig = plt.gcf()
  
  # Adding Circle in Pie chart
  fig.gca().add_artist(centre_circle)
  

  buf = BytesIO()
  fig.savefig(buf, format="png")
  st.image(buf)



Category=  pd.read_csv(r"Category.csv")

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
                    summary = content.find('div',class_ = 'readable stacked')
                    summary_ = summary.text

                    book_data_st = pd.DataFrame({'summary': [summary_[:-8]]}) 
    
    
                    
              
                    with st.expander("Book Summary"):
                        st.info(summary_[:-8])
    
    
                    with st.expander("Tell me the overall sentiment of this book"):
                        find_sentiment(book_data_st.summary, book_data_st.summary.values[0])
                      
                      
                    with st.expander("Show me the detailed sentiment analysis"):
                        pie_chart(book_data_st.summary.values[0])
                    

                    with st.expander("Book URL"):
                     
                      st.caption(url)
              
    
                   
                  

                except:
                    st.caption('Sorry We are Unable to Find Summary For this Book')    
