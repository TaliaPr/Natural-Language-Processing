import spacy

class data:
    def __init__(self, data):
        self._words = []
        self._pos = []
        self._bio = []
        for sentence in data:
            for word, pos, bio in sentence:
                self._words.append(word)
                self._pos.append(pos)
                self._bio.append(bio)

    def __call__(self, language="es"):
        # Load the Spanish model if not already loaded
        if language == "es":
            nlp = spacy.load("es_core_news_sm")
        elif language == "nl":
            nlp = spacy.load("nl_core_news_sm")
        
        # Set a larger max_length if needed (but be careful with memory)
        nlp.max_length = 1500000  # Increase if you're sure your system can handle it
        
        # Process in chunks if text is too large
        chunk_size = 100000  # characters per chunk
        lemmas = []
        
        # Split words into chunks and process each chunk
        current_chunk = []
        current_length = 0
        
        for word in self._words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > chunk_size:
                # Process current chunk
                chunk_text = " ".join(current_chunk)
                doc = nlp(chunk_text)
                chunk_lemmas = [token.lemma_ for token in doc]
                lemmas.extend(chunk_lemmas)
                
                # Start new chunk
                current_chunk = [word]
                current_length = word_len
            else:
                # Add to current chunk
                current_chunk.append(word)
                current_length += word_len
        
        # Process the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            doc = nlp(chunk_text)
            chunk_lemmas = [token.lemma_ for token in doc]
            lemmas.extend(chunk_lemmas)
        
        # Ensure we have the same number of lemmas as words
        if len(lemmas) != len(self._words):
            # In case of mismatch (due to tokenization differences), fallback to simple lemmatization
            lemmas = []
            for word in self._words:
                doc = nlp(word)
                lemmas.append(doc[0].lemma_)
        
        return lemmas

    def get_word_pos(self):
        return list(zip(self._words, self._pos))
    
    def get_word_bio(self):
        return list(zip(self._words, self._bio))
    
    def get_word(self):
        return self._words
    def get_pos(self):
        return self._pos
    def get_bio(self):
        return self._bio
    
class test_data:
    def __init__(self, data):
        self._words = []
        self._pos = []
        self._bio = []
        for word, pos, bio in data:
            self._words.append(word)
            self._pos.append(pos)
            self._bio.append(bio)
    
    def get_word_pos(self):
        return list(zip(self._words, self._pos))
    
    def get_word_bio(self):
        return list(zip(self._words, self._bio))