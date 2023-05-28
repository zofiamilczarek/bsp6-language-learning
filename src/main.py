import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from level_prediction.utils.preprocess import Preprocessing as prep
from level_prediction.utils.features import get_text_features
from text_recommender.text_recommender import recommend_text
import os
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
abs_path_prefix = os.path.dirname(os.path.abspath(__file__)) + '/'

def load_bundle(locale):
    df = pd.read_csv(abs_path_prefix+"language_locale.csv")
    df = df.query(f"locale == '{locale}'")
    lang_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict

def get_recommendation(text,lang_dict):
    try:
        recommendation = recommend_text(text, user_level)
    except:
        recommendation = lang_dict['recommend_error']

    return handle_markdown_special_chars(recommendation)

def display_recommendation(lang_dict):
    st.subheader(lang_dict['recommend_header'])
    recommended_text  = get_recommendation(text,lang_dict)
    st.markdown(recommended_text)

st.set_page_config(page_title="English Learning App",
            page_icon=":books:",
            layout='wide')
side_1, side_2 = st.sidebar.columns(2)


lang_options = {
        "English":"en_US",
        "Polski":"pl_PL"
    }

with side_2:
    locale = st.radio(label='Language', options=list(lang_options.keys()))

lang_dict = load_bundle(lang_options[locale])

st.title('Let\'s learn English!')
st.markdown(handle_markdown_special_chars(lang_dict['description']).replace('\\n', '  \n'))

model_level = pkl.load(open(abs_path_prefix + 'level_prediction/models/svm.pkl', 'rb'))


#create a box with text input where a user can enter a long form text

text = st.text_area(lang_dict['type_here'])
try:
    #get the level of the text
    vectorized_text = get_text_features(text,with_pos=False)
    user_level = model_level.predict(vectorized_text)
    user_level = prep.decode_label(user_level[0])
    prediction_msg = lang_dict['level_message'] + str(user_level)
except ValueError as e:
    prediction_msg = lang_dict['level_error']
    print(e)


if 'generate_text' not in st.session_state:
    st.session_state.generate_text = False

def change_generation():
    st.session_state.generate_text = not st.session_state.generate_text

if not st.session_state.generate_text:
    st.button(lang_dict['generate_text'],on_click=change_generation)

if st.session_state.generate_text:    
    st.markdown(prediction_msg)
    regenerate_button = st.button(lang_dict['generate_another'])
    display_recommendation(lang_dict)
    if regenerate_button:
        st.session_state.generate_text = True

