import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from level_prediction.utils.preprocess import Preprocessing as prep
from level_prediction.utils.features import get_text_features
from text_recommender.text_recommender import recommend_text

import pickle

def handle_markdown_special_chars(text):
    text = text.replace('#', '\#')
    text = text.replace('-', '\-')
    text = text.replace('*', '\*')
    text = text.replace('_', '\_')
    text = text.replace('`', '\`')
    text = text.replace('$', '\$')
    text = text.replace('\n', '  \n')
    return text

def get_recommendation(text):
    try:
        recommendation = recommend_text(text, user_level)
    except:
        recommendation = 'We could not find a recommended text for you. Try again!'

    return handle_markdown_special_chars(recommendation)

def display_recommendation():
    st.subheader('Recommended text')
    recommended_text  = get_recommendation(text)
    st.markdown(recommended_text)


st.title('Let\'s learn English!')
st.markdown('This is a tool that will help you find the right text for your level of English. \nJust enter your text and we will recommend you a text. \nYou can write about anything you want to, but we recommend that you write about something that you are interested in. \nWe will also tell you the level of your text. \nHave fun!')
model_level = pkl.load(open('level_prediction/models/svm.pkl', 'rb'))

#create a box with text input where a user can enter a long form text
text = st.text_area('Type here')
try:
    #get the level of the text
    vectorized_text = get_text_features(text,with_pos=False)
    user_level = model_level.predict(vectorized_text)
    user_level = prep.decode_label(user_level[0])
    prediction_msg = 'The level of your text is: ' + str(user_level)
except:
    prediction_msg = 'You entered empty text!'


if 'generate_text' not in st.session_state:
    st.session_state.generate_text = False

def change_generation():
    st.session_state.generate_text = not st.session_state.generate_text


st.button('Generate text',on_click=change_generation)

if st.session_state.generate_text:    
    st.markdown(prediction_msg)
    regenerate_button = st.button('Generate another text')
    display_recommendation()
    if regenerate_button:
        submit_button = True

