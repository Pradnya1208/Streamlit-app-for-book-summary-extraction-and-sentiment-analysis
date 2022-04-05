# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 13:16:44 2022

@author: Pradnya Patil
"""

# -*- coding: utf-8 -*-
#from tracemalloc import stop
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup as bs
import requests

import data_explorer
import sentiment_analyzer
import streamlit as st

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black;'>Summary Extraction and Sentiment Analysis</h1>", unsafe_allow_html=True)

PAGES = {
    "Sentiment Analysis": sentiment_analyzer,
    "Data Exploration": data_explorer
    
}
#st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.sentiment_wrapper()











