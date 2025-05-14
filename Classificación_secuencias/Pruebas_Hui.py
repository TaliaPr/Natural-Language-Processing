import spacy
import nltk
from nltk.corpus import conll2002
import spacy
import matplotlib.pyplot as plt
import string
from nltk.tag import CRFTagger
from itertools import combinations
import time
import pandas as pd
import seaborn as sns
train = conll2002.iob_sents('esp.train')


class data:
    def __init__(self, data, language="es"):
        self._data = data
        self._words = []
        self._pos = []
        self._bio = []
        for sentence in data:
            for word, pos, bio in sentence:
                self._words.append(word)
                self._pos.append(pos)
                self._bio.append(bio)
        
        if language == "es":
            self._nlp = spacy.load("es_core_news_sm")
        elif language == "nl":
            self._nlp = spacy.load("nl_core_news_sm")
        else:
            raise ValueError("Unsupported language")
        
        self._nlp.max_length = 1500000
        self.language = language
    def lemmatize(self):
        lemmas = []
        # Process each sentence individually
        for sentence in self._data:
            # Extract just the words for lemmatization
            words = [word for word, _, _ in sentence]
            text = " ".join(words)
            
            # Process with SpaCy
            doc = self._nlp(text)
            
            # Extract lemmas and add to the list
            for token in doc:
                lemmas.append(token.lemma_)
                
        return lemmas

    def get_word_pos(self):
        return list(zip(self._words, self._pos))
    
    def get_word_bio(self):
        return list(zip(self._words, self._bio))
    def get_all_lemmas(self):
        processed_sentences = []
        
        for sentence in self._data:
            # Extract words for this sentence
            words = [word for word, _, _ in sentence]
            text = " ".join(words)
            
            # Process with SpaCy
            doc = self._nlp(text)
            
            # Create processed sentence
            processed_sentence = []
            for i, (word, pos, bio) in enumerate(sentence):
                if i < len(doc):
                    lemma = doc[i].lemma_
                    processed_sentence.append(((word, pos, lemma), bio))
                else:
                    # Fallback in case of token mismatch
                    processed_sentence.append(((word, pos, word.lower()), bio))
            
            processed_sentences.append(processed_sentence)
            
        return processed_sentences
    
    def get_word(self):
        return self._words
    def get_pos(self):
        return self._pos
    def get_bio(self):
        return self._bio
    
train = data(train)
print("===========================================================")
train_lemmas = list(train.get_all_lemmas())
print(train_lemmas[0:10])
