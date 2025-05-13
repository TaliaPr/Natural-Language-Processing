

class ner:
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
    
    
