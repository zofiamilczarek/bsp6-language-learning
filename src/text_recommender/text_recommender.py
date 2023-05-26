import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = pickle.load(open('text_recommender/embeddings.pkl', 'rb'))
texts = pd.read_csv('text_recommender/datasets/cefr_texts_labeled.csv')

def recommend_text(text, level, embeddings=embeddings, model=model, texts=texts):
    """
    Returns a list of 3 recommended texts based on the input text.
    """
    
    level_up = {'A1': 'A2', 'A2': 'B1', 'B1': 'B2', 'B2': 'C1', 'C1': 'C2', 'C2': 'C2'}

    #get the indices with the same or higher level as the input text
    level_indices = (texts[texts['label'] == level] + texts[texts['label'] == level_up[level]]).index
    embeddings = embeddings[level_indices]

    #embed the input text
    text_embedding = model.encode(text)
    text_embedding = np.array(text_embedding).reshape(1, -1)

    #calculate the cosine similarity between the input text and all the texts in the dataset
    cos_sim_data = pd.DataFrame(cosine_similarity(text_embedding,embeddings))
    #get the indices of the 3 most similar texts
    index_recomm =cos_sim_data.loc[0].sort_values(ascending=False).index.tolist()[0:6]
    #get the recommended texts
    recommended_texts = texts.iloc[index_recomm]['text'].values
    random_index = np.random.randint(0, len(recommended_texts))
    return recommended_texts[random_index]

if __name__ == '__main__':
    print(recommend_text('I like to play football.', 'B2'))