
from nltk import pos_tag, word_tokenize, sent_tokenize, download
import spacy
import numpy as np
import textstat

download('punkt')

nlp = spacy.load('en_core_web_sm', disable=['ner'])

def get_pos_frequencies(sentence):
    pos_tags = pos_tag(word_tokenize(sentence))
    pos_frequencies = {}
    for tag in pos_tags:
        if tag[1] in pos_frequencies:
            pos_frequencies[tag[1]] += 1
        else:
            pos_frequencies[tag[1]] = 1
    return pos_frequencies

def get_mean_pos(text):
    sentences = [get_pos_frequencies(sent) for sent in text]
    nr_of_sentences = len(sentences)
    mean_pos = {}
    for sent in sentences:
        for pos in sent.keys():
            if pos in mean_pos.keys():
                mean_pos[pos]+=1
            else:
                mean_pos[pos] = 1
    mean_pos = dict(map(lambda x : (x[0], x[1]/nr_of_sentences), mean_pos.items()))

    return mean_pos

def tree_depth(root):
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_depth(x) for x in root.children)
    
def get_avg_parse_tree_depth(text):
    doc = nlp(text)
    roots = [sent.root for sent in doc.sents]
    return (np.mean([tree_depth(root) for root in roots]),[tree_depth(root) for root in roots])

def get_text_features(text,with_pos=False):
    features= {
        'length': len(text),
        'num_words': len(text.split(" ")),
        'num_sentences': len(sent_tokenize(text)),
        'avg_word_length': np.mean([len(word) for word in text.split(" ")]),
        'avg_sentence_length': np.mean([len(sentence.split(" ")) for sentence in text.split(".")]),
        'smog': textstat.smog_index(text),
        'flesch': textstat.flesch_reading_ease(text),
        'dalle-chall': textstat.dale_chall_readability_score(text),
        'difficult_words': textstat.difficult_words(text),
        'linsear': textstat.linsear_write_formula(text),
        'text_standard': textstat.text_standard(text,float_output=True),
        'readability':textstat.automated_readability_index(text),
        'avg_parse_tree': get_avg_parse_tree_depth(text)[0],
        'max_parse_tree': max(get_avg_parse_tree_depth(text)[1]),
        'min_parse_tree': min(get_avg_parse_tree_depth(text)[1]),
    }
    if with_pos:
        features.update(get_mean_pos(text))
    return features