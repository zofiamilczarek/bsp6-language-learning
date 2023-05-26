import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk import download
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, stem_text, strip_punctuation
import os
#download('stopwords')

CEFR_LEVELS = ["A1","A2","B1","B2","C1","C2"]
LEMMATIZER = WordNetLemmatizer()
class Preprocessing():
    def load_data(filename):
        path = filename
        dataset = pd.read_csv(path)
        return dataset

    def data_exploration(dataframe):
        labels = dataframe.groupby(['label']).count()

    def tokenize_and_lemmatize(text):
        "Transforms a dataset from texts labeled with strings into lists of lemmatized words with numerical labels"
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        words = word_tokenize(text)
        processed_text = [LEMMATIZER.lemmatize(w) for w in words if not w in stopwords.words()] 
        return " ".join(processed_text)


    def clean(texts, should_stem=True, should_remove_punctuation=True, should_remove_stopwords=True):
        filters = [lambda x : x.lower()]
        if should_stem:
            filters.append(stem_text)
        if should_remove_punctuation:
            filters.append(strip_punctuation)
        if should_remove_stopwords:
            filters.append(remove_stopwords)
    
        clean_texts = texts.apply(lambda x: " ".join(preprocess_string(x, filters)))
        return clean_texts
    


    def apply_tfidf(texts):
        #text_frame["text"] = text_frame["text"].apply(lambda x: Preprocessing.tokenize_and_lemmatize(x))
        #filters = [lambda x : x.lower(), remove_stopwords, strip_tags, stem_text]
        #text = texts.apply(lambda x: preprocess_string(x, filters))
        tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        ngram_range=(1, 3), 
                        stop_words='english')
        vectorized = tfidf.fit_transform(texts).toarray()
        return vectorized
    
    def apply_word2vec(texts):
        pass
    

    def encode_label(label):
        label2id = {'A1':1,'A2':2,'B1':3,'B2':4,'C1':5,'C2':6}
        new_label = label2id[label]
        return new_label
    
    def decode_label(label):
        label2id = {'A1':1,'A2':2,'B1':3,'B2':4,'C1':5,'C2':6}
        id2label = {key:id for id,key in label2id.items()}
        new_label = id2label[label]
        return new_label
    
    def encode_labels_simplified(label):
        label2id = {'A1':1,'A2':1,'B1':2,'B2':2,'C1':3,'C2':4}
        new_label = label2id[label]
        return new_label
    
def map_label_to_CEFR(label):
    if "+" in label:
        return label.replace("+","")
    elif "-" in label:
        if "<" in label:
            label = label.replace("<","")
        return label.split("-")[0]
    else:
        return label


def transform_cefr_txt_dataset(path):
    dataset = pd.DataFrame(columns=["text","label"])
    for subdir in os.listdir(path):
        print(subdir)
        if subdir.endswith(".DS_Store"):
            continue
        for filename in os.listdir(os.path.join(path,subdir)):
            if filename.endswith(".txt"):
                with open(os.path.join(path,subdir,filename), 'r') as f:
                    text = f.read()
                    label = map_label_to_CEFR(subdir)
                    dataset = pd.concat([dataset,pd.DataFrame({"text":[text],"label":[label]})],ignore_index=True)
            else:
                continue

    return dataset



if __name__=="__main__":
    d = transform_cefr_txt_dataset("datasets/cefr_text_dataset/en")
    print(d.head())
    print(d.groupby("label").count())
    print(d.shape)
    d.to_csv("datasets/cefr_leveled_texts_2.csv",index=False)

