import string
import nltk
from nltk.corpus import conll2002
import spacy
import matplotlib.pyplot as plt
from nltk.tag import CRFTagger
from itertools import combinations
from collections import Counter
from typing import Dict, List, Tuple, Union, Set, Any, Optional, Callable
import time
import pandas as pd
import seaborn as sns
import numpy as np


# Crear una función para graficar
def plot_tag_distribution(tag_counts: Dict[str, int], title: str, exclude_tag: Optional[str] = None) -> None:
    """ Función para graficar la distribución de etiquetas en el conjunto de entrenamiento.
    :param tag_counts: Contador de etiquetas
    :param title: Título del gráfico
    :param exclude_tag: Etiqueta a excluir del gráfico
    """
    if exclude_tag:
        tag_counts = {tag: count for tag, count in tag_counts.items() if tag != exclude_tag}
    plt.figure(figsize=(5, 3))
    plt.bar(tag_counts.keys(), tag_counts.values(), color='skyblue')
    plt.xlabel('Etiquetas')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


class SimpleGazetteerExtractor:
    """
    Clase para extraer patrones de entidades LOC, ORG, y MISC y analizar trigramas precedentes.
    """
    def __init__(self) -> None:
        self.loc_patterns = Counter()
        self.org_patterns = Counter()
        self.misc_patterns = Counter()
        self.trigrams_before_org = Counter()
        self.trigrams_before_loc = Counter()
        self.trigrams_before_misc = Counter()

    def extract_patterns_and_trigrams(self, training_data: List[Tuple[List[str], List[str]]], corpus: List[List[Tuple[str, str, str]]]) -> None:
        """
        Extrae patrones de entidades LOC, ORG, y MISC y analiza trigramas precedentes.

        Args:
            training_data: Lista de ejemplos donde cada ejemplo es una tupla (tokens, tags).
            corpus: Lista de oraciones con tokens, etiquetas POS y etiquetas NER.
        """
        # Extraer patrones
        for sentence, tags in training_data:
            current_entity = []
            current_type = None

            for token, tag in zip(sentence, tags):
                if tag.startswith('B-'):
                    if current_entity and current_type:
                        self._store_entity(' '.join(current_entity), current_type)
                    current_entity = [token]
                    current_type = tag[2:]
                elif tag.startswith('I-') and current_type == tag[2:]:
                    current_entity.append(token)
                else:
                    if current_entity and current_type:
                        self._store_entity(' '.join(current_entity), current_type)
                    current_entity = []
                    current_type = None

            if current_entity and current_type:
                self._store_entity(' '.join(current_entity), current_type)

        # Analizar trigramas precedentes
        self._analyze_trigrams(corpus)

    def _store_entity(self, entity: str, entity_type: str) -> None:
        """Almacena la entidad en el contador correspondiente."""
        if entity_type == 'LOC':
            self.loc_patterns[entity] += 1
        elif entity_type == 'ORG':
            self.org_patterns[entity] += 1
        elif entity_type == 'MISC':
            self.misc_patterns[entity] += 1

    def _analyze_trigrams(self, corpus: List[List[Tuple[str, str, str]]]) -> None:
        """Analiza los trigramas que preceden a organizaciones, lugares y misceláneos."""
        for sentence in corpus:
            for i in range(len(sentence)):
                _, _, ner_tag = sentence[i]

                if ner_tag == 'B-ORG' and i >= 3:
                    trigram = (
                        sentence[i-3][0],
                        sentence[i-2][0],
                        sentence[i-1][0]
                    )
                    self.trigrams_before_org[trigram] += 1

                if ner_tag == 'B-LOC' and i >= 3:
                    trigram = (
                        sentence[i-3][0],
                        sentence[i-2][0],
                        sentence[i-1][0]
                    )
                    self.trigrams_before_loc[trigram] += 1

                if ner_tag == 'B-MISC' and i >= 3:
                    trigram = (
                        sentence[i-3][0],
                        sentence[i-2][0],
                        sentence[i-1][0]
                    )
                    self.trigrams_before_misc[trigram] += 1

    def print_patterns_and_trigrams(self) -> None:
        """Imprime los patrones y trigramas más comunes encontrados."""
        print("=== Top 30 Patrones de LOC ===")
        for entity, freq in self.loc_patterns.most_common(30):
            print(f"{entity}: {freq}")

        print("\n=== Top 30 Patrones de ORG ===")
        for entity, freq in self.org_patterns.most_common(30):
            print(f"{entity}: {freq}")

        print("\n=== Top 30 Patrones de MISC ===")
        for entity, freq in self.misc_patterns.most_common(30):
            print(f"{entity}: {freq}")

        print("\n=== Top 30 Trigramas que preceden a organizaciones ===")
        for trigram, freq in self.trigrams_before_org.most_common(30):
            print(f"{trigram[0]} {trigram[1]} {trigram[2]}: {freq}")

        print("\n=== Top 30 Trigramas que preceden a lugares ===")
        for trigram, freq in self.trigrams_before_loc.most_common(30):
            print(f"{trigram[0]} {trigram[1]} {trigram[2]}: {freq}")

        print("\n=== Top 30 Trigramas que preceden a misceláneos ===")
        for trigram, freq in self.trigrams_before_misc.most_common(30):
            print(f"{trigram[0]} {trigram[1]} {trigram[2]}: {freq}")

class OptimizedFeatFunc:
    def __init__(self, use_Basic: bool = True, use_context_words: bool = True, use_contex_POS_tag: bool = True, use_specific_caracteristics: bool = True, use_lemas: bool = True, use_EXTRA: bool = False) -> None:
        """
        Constructor de la clase de las funciones de características para el CRFTagger.
        Uso:
        - use_Basic: Si se deben usar características básicas (longitud, mayúsculas, etc.)
        - use_context_words: Si se deben usar palabras de contexto (palabra anterior y siguiente)
        - use_contex_POS_tag: Si se deben usar etiquetas POS de contexto (etiqueta anterior y siguiente)
        - use_specific_caracteristics: Si se deben usar características específicas (Gazetteer)
        - use_lemas: Si se deben usar lemas (forma base de la palabra)
        - use_extra: Gazzetters para el apartado opcional CADEC Corpus. 

        Importante: Si use_extra está en True, entonces el usuario debe fijar use_specific_caracteristics a False 
       
        """
        self.use_basic_features = use_Basic
        self.use_context = use_context_words
        self.use_conext_POS_tags = use_contex_POS_tag
        self.use_specific_caracteristics = use_specific_caracteristics
        self.use_lema = use_lemas
        self.use_extra = use_EXTRA
        
    
        self.ciudades = {"Madrid", "España", "Barcelona", "París", "Líbano", "Badajoz", "Santander", "Mérida", "Cáceres", "Europa", "Brasil", "Cataluña"}
        self.organizaciones_comunes = {"EFE", "Gobierno", "PP", "EFECOM", "UE", "Real Madrid", "Efe",  "PSOE", "Estado", "Ejército", "Congreso",  "Ejecutivo", "Ayuntamiento", "ESL", "Telefónica", "ONU", "IU", "Unión Europea", "CiU", "Guardia Civil", "Hizbulá", "ETA"}

        self.common_org_precedents = {
            "presidente de la",
            "El presidente del"
            "el presidente del",
            "el presidente de",
            "El presidente de"
            "el portavoz del",
            "El portavoz del",
            "general de la",
            "fuentes de la",
            "director general de",
            'miembros de la',
            'representantes de la',
            'secretario general de'
        }

        
        # Cache para resultados
        self.cache = {}


        
        # Common pain/symptom words
        self.symptom_words = {
            'pain', 'ache', 'sore', 'nausea', 'fatigue', 'tired', 'dizzy', 'headache',
            'migraine', 'cramp', 'spasm', 'numbness', 'tingling', 'rash', 'itch', 'swelling',
            'inflammation', 'burning', 'stiff', 'stiffness', 'weak', 'weakness', 'blur', 'blurry'
        }
        
        # Common finding/observation words
        self.finding_words = {
            'elevated', 'high', 'low', 'increase', 'decrease', 'normal', 'abnormal', 'positive',
            'negative', 'acute', 'chronic', 'intermittent', 'constant', 'severe', 'mild', 'moderate'
        }
        
        # Common medication brand names
        self.drug_names = {
            'lipitor', 'crestor', 'zocor', 'pravachol', 'voltaren', 'advil', 'tylenol',
            'ibuprofen', 'aspirin', 'acetaminophen', 'naproxen', 'atorvastatin', 'rosuvastatin',
            'simvastatin', 'pravastatin', 'diclofenac', 'pennsaid', 'tricor', 'ezetrol', 'arthrotec'
        }
        

    def __call__(self, tokens: List[Tuple[str, str, str]], idx: int) -> Dict[str, Any]:
        # Obtener la clave única para la oración actual
        sentence_key = tuple(tokens) # se convierte a tupla porque las tuplas son inmutables
        
        # Clave única para caché
        cache_key = (sentence_key, idx)
        
        # Verificar si ya calculamos este caso
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        feats = {}
        
        if idx >= len(tokens) or idx < 0:
            self.cache[cache_key] = feats
            return feats
            
        word = tokens[idx][0]

        if self.use_basic_features:
            # Características básicas
            feats["word"] = word
            feats["length"] = len(word)
            
            # Características calculadas una sola vez
            if any(char.isdigit() for char in word):
                feats["has_number"] = True
                
            if word and word[0].isupper():
                feats["is_capitalized"] = True
                
            if any(c in string.punctuation for c in word):
                feats["punctuated"] = True
                
            if len(word) > 1:
                feats["suffix"] = word[-2:]
                feats['prefix'] = word[:2]
                
            if word and word.isupper():
                feats['all_capital'] = True

        if self.use_context:
            feats["word"] = word
            if idx >= 1 and idx < len(tokens)-1:
                feats["prev_word"] = tokens[idx-1][0]
                feats["next_word"] = tokens[idx+1][0]


        if self.use_conext_POS_tags:
            
            pos = tokens[idx][1]
            feats["POS"] = pos

            if idx >= 1 and idx < len(tokens)-1:
                
                # Crear trigramas eficientemente
                prev_pos = tokens[idx-1][1]
                next_pos = tokens[idx+1][1]
                
                prev_prev_pos = tokens[idx-2][1] if idx >= 2 else 'START'
                next_next_pos = tokens[idx+2][1] if idx < len(tokens)-2 else 'END'
                
                feats["Prev_POS_2"] = prev_prev_pos
                feats["Prev_POS_1"] = prev_pos
                feats["next_POS_1"] = next_pos
                feats["next_pos_2"] = next_next_pos
                
        if self.use_lema:
           
            lem = tokens[idx][2]
            feats["lema"] = lem
            
            
        if self.use_specific_caracteristics:
            
            if word in self.ciudades:
                feats["is_city"] = True
            
            if word in self.organizaciones_comunes:
                feats["is_organization"] = True
        
            # Verificar precedentes de organización
            if idx >= 3:
                word1 = tokens[idx-3][0]
                word2 = tokens[idx-2][0]
                word3 = tokens[idx-1][0]

                if word1.isdigit() and word2.isalpha() and word3 == "(":
                    feats["organization_precedent"] = True
                else:
                    precedent = f"{word1} {word2} {word3}".lower()
                    if precedent in self.common_org_precedents:
                        feats["organization_precedent"] = True

        if self.use_extra:
            
            if word in self.symptom_words:
                feats["symptom_word"] = True
            
            if word in self.finding_words:
                feats["finding_word"] = True

            if word in self.drug_names:
                feats["drug_name"] = True

        # Guardar en caché
        self.cache[cache_key] = feats
        return feats
    


# Load SpaCy model for Spanish
nlp = spacy.load("es_core_news_sm")

def prepare_data_for_crf(conll_data: List[List[Tuple[str, str, str]]], include_lemmas: bool = True) -> List[List[Tuple[Tuple[str, str, str], str]]]:
    """Process conll data into format for CRF tagging with optional lemmatization.
    
    Args:
        conll_data: List of sentences, where each sentence is a list of tuples (word, pos, tag).
        include_lemmas: Whether to include lemmatization in the processed data.
        
    Returns:
        Processed data in the format required for CRF tagging, that is where each tuple has two arguments.
        In the first argument, the word, pos and lemma (optional) are included, and in the second argument the tag.
    """
    processed_data = []
    
    for sentence in conll_data:
        # Process entire sentence for better lemmatization context
        if include_lemmas:
            text = " ".join(word for word, _, _ in sentence)#works better using the whole sentence (Spacy needs context)
            doc = nlp(text) 
            
            # Create processed sentence
            processed_sentence = []
            for i, (word, pos, tag) in enumerate(sentence):
                if i < len(doc):
                    lemma = doc[i].lemma_
                    processed_sentence.append(((word, pos, lemma), tag))
                else:
                    # Fallback in case of token mismatch
                    processed_sentence.append(((word, pos, word), tag))
        else:
            processed_sentence = [((word, pos), tag) for word, pos, tag in sentence]
            
        processed_data.append(processed_sentence)
    
    return processed_data


def extract_BIO(tags: List[Union[str, Tuple[Any, str]]]) -> List[Tuple[str, int, int]]:
    entities = []
    entity_type = None
    start_idx = None
    
    for i, tag in enumerate(tags):
        # Handle the case where tag might be a tuple
        if isinstance(tag, tuple):
            tag = tag[1]  # Extract the actual tag if it's a tuple (word, tag)
            
        if tag.startswith('B-'):
            # If we were tracking an entity, add it to the list
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
            # Start a new entity
            entity_type = tag[2:]  # Remove 'B-' prefix
            start_idx = i
        elif tag.startswith('I-'):
            # Continue with the current entity
            curr_type = tag[2:]  # Remove 'I-' prefix
            # This handles inconsistent I- tags that don't match the current entity
            if entity_type is None or curr_type != entity_type:
                # Close any open entity and ignore this tag (it's an error in tagging)
                if entity_type is not None:
                    entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
        else:  # 'O' tag
            # If we were tracking an entity, add it to the list
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
    
    # Don't forget the last entity if the sequence ends with an entity
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
    
    return entities

def extract_IO(tags: List[Union[str, Tuple[Any, str]]]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from IO tagging scheme.
    
    Args:
        tags: List of IO tags (e.g., 'I-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """
    entities = []
    entity_type = None
    start_idx = None
    
    for i, tag in enumerate(tags):
        # Handle the case where tag might be a tuple
        if isinstance(tag, tuple):
            tag = tag[1]  # Extract the actual tag if it's a tuple (word, tag)
            
        if tag.startswith('I-'):
            current_type = tag[2:]  # Remove 'I-' prefix
            
            # If not currently tracking an entity OR the entity type changes
            if entity_type is None:
                entity_type = current_type
                start_idx = i
            elif current_type != entity_type:
                # Close current entity and start a new one
                entities.append((entity_type, start_idx, i - 1))
                entity_type = current_type
                start_idx = i
        else:  # 'O' tag
            # If we were tracking an entity, add it to the list
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
    
    # Don't forget the last entity if the sequence ends with an entity
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
    
    return entities

def extract_BIOE(tags: List[Union[str, Tuple[Any, str]]]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from BIOE tagging scheme.
    
    Args:
        tags: List of BIOE tags (e.g., 'B-PER', 'I-PER', 'E-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """
    entities = []
    entity_type = None
    start_idx = None
    
    for i, tag in enumerate(tags):
        # Handle the case where tag might be a tuple
        if isinstance(tag, tuple):
            tag = tag[1]  # Extract the actual tag if it's a tuple
            
        if tag.startswith('B-'):
            # If we were tracking an entity, add it to the list (should not happen in valid BIOE)
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
            
            # Start a new entity
            entity_type = tag[2:]  # Remove 'B-' prefix
            start_idx = i
        elif tag.startswith('I-'):
            # Continue with the current entity - no change needed
            pass
        elif tag.startswith('E-'):
            # End current entity and add it to list
            if entity_type is not None and entity_type == tag[2:]:
                entities.append((entity_type, start_idx, i))
                entity_type = None
                start_idx = None
            else:
                # Handle error case: E- without matching B-
                entities.append((tag[2:], i, i))
        else:  # 'O' tag
            # If we were tracking an entity, add it to the list (should not happen in valid BIOE)
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
    
    # Check for incomplete entity at end (should not happen in valid BIOE)
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
    
    return entities

def extract_BIOES(tags: List[Union[str, Tuple[Any, str]]]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from BIOES tagging scheme.
    
    Args:
        tags: List of BIOES tags (e.g., 'B-PER', 'I-PER', 'E-PER', 'S-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """
    entities = []
    entity_type = None
    start_idx = None
    
    for i, tag in enumerate(tags):
        # Handle the case where tag might be a tuple
        if isinstance(tag, tuple):
            tag = tag[1]  # Extract the actual tag if it's a tuple
            
        if tag.startswith('B-'):
            # If we were tracking an entity, add it to the list (should not happen in valid BIOES)
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
            
            # Start a new entity
            entity_type = tag[2:]  # Remove 'B-' prefix
            start_idx = i
        elif tag.startswith('I-'):
            # Continue with the current entity - no change needed
            pass
        elif tag.startswith('E-'):
            # End current entity and add it to list
            if entity_type is not None and entity_type == tag[2:]:
                entities.append((entity_type, start_idx, i))
                entity_type = None
                start_idx = None
            else:
                # Handle error case: E- without matching B-
                entities.append((tag[2:], i, i))
        elif tag.startswith('S-'):
            # Single token entity
            entities.append((tag[2:], i, i))
        else:  # 'O' tag
            # If we were tracking an entity, add it to the list (should not happen in valid BIOES)
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
    
    # Check for incomplete entity at end (should not happen in valid BIOES)
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
    
    return entities

def extract_BIOW(tags: List[Union[str, Tuple[Any, str]]]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from BIOW tagging scheme.
    
    Args:
        tags: List of BIOW tags (e.g., 'B-PER', 'I-PER', 'W-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """
    entities = []
    entity_type = None
    start_idx = None
    
    for i, tag in enumerate(tags):
        # Handle the case where tag might be a tuple
        if isinstance(tag, tuple):
            tag = tag[1]  # Extract the actual tag if it's a tuple
            
        if tag.startswith('B-'):
            # If we were tracking an entity, add it to the list (should not happen in valid BIOW)
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
            
            # Start a new entity
            entity_type = tag[2:]  # Remove 'B-' prefix
            start_idx = i
        elif tag.startswith('I-'):
            # Continue with the current entity - no change needed
            pass
        elif tag.startswith('W-'):
            # Whole entity (single token)
            entities.append((tag[2:], i, i))
        else:  # 'O' tag
            # If we were tracking an entity, add it to the list
            if entity_type is not None:
                entities.append((entity_type, start_idx, i - 1))
                entity_type = None
                start_idx = None
    
    # Check for incomplete entity at end
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
    
    return entities


def extract_entities_types(gold_entities: List[Tuple[str, int, int]], pred_entities: List[Tuple[str, int, int]]) -> Set[str]:
    entity_types = set()
    # Add entity types to our set
    for entity_type, _, _ in gold_entities:
        entity_types.add(entity_type)
    for entity_type, _, _ in pred_entities:
        entity_types.add(entity_type)
    return entity_types  # Added return statement

def extract_confusion_matrix(gold_entities: List[Tuple[str, int, int]], pred_entities: List[Tuple[str, int, int]]) -> Dict[Tuple[str, str], int]:
    """
    Extract confusion matrix for entity recognition.
    
    Args:
        gold_entities: List of gold standard entity tuples (type, start, end)
        pred_entities: List of predicted entity tuples (type, start, end)
        
    Returns:
        Dictionary representing the confusion matrix
    """
    confusion_matrix = {}
    # Build confusion matrix
    
    # Track positions that have been processed
    processed_positions = set()
    
    # For each gold entity, find if it was correctly predicted
    for gold_entity in gold_entities:
        gold_type, start, end = gold_entity
        
        # Look for predicted entity at the same position
        matched = False
        for pred_entity in pred_entities:
            pred_type, p_start, p_end = pred_entity

            if start == p_start and end == p_end:  # Check exact match
                # Update confusion matrix
                confusion_key = (gold_type, pred_type)
                confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
                matched = True
                # Mark this position as processed
                processed_positions.add((p_start, p_end))
                break
        
        # If no matching prediction was found, it's a false negative
        if not matched:
            confusion_key = (gold_type, "O")  # "O" represents no prediction
            confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
    
    # Process false positives (predictions without gold)
    for pred_entity in pred_entities:
        pred_type, p_start, p_end = pred_entity
        # Check if this prediction has already been processed (matched with gold)
        if (p_start, p_end) not in processed_positions:
            confusion_key = ("O", pred_type)  # "O" represents no gold entity
            confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
            
    # Add true negatives count (O,O) - this requires knowing the total number of tokens
    # that should be labeled as "O" in both gold and predicted data
    # We need to add additional information to correctly count (O,O) cases
    
    return confusion_matrix

def extract_entities(tags: List[Union[str, Tuple[Any, str]]], otherTAG: Optional[str] = None) -> List[Tuple[str, int, int]]:
    """
    Extract entity spans from a sequence of BIO tags.
    
    Args:
        tags: List of BIO tags (e.g., 'B-PER', 'I-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """

    if not otherTAG:
        entities = extract_BIO(tags)

    elif otherTAG == 'IO':
        entities = extract_IO(tags)

    elif otherTAG == 'BIOE':
        entities = extract_BIOE(tags)

    elif otherTAG == 'BIOES':
        entities = extract_BIOES(tags)
    
    elif otherTAG == 'BIOW':
        entities = extract_BIOW(tags)

    return entities
    


def evaluate_entities(gold_entities: List[Tuple[str, int, int]], pred_entities: List[Tuple[str, int, int]]) -> Dict[str, int]:
    """
    Calculate precision, recall, and F1 score for entity recognition.
    
    Args:
        gold_entities: List of gold standard entity tuples (type, start, end)
        pred_entities: List of predicted entity tuples (type, start, end)
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Convert to sets for easier comparison
    gold_set = set(gold_entities)
    pred_set = set(pred_entities)
    
    # Calculate correct predictions (intersection)
    correct = len(gold_set.intersection(pred_set))
    
    return {
        'gold_count': len(gold_set),
        'pred_count': len(pred_set),
        'correct': correct
    }

def evaluate_ner_corpus(gold_data: List[List[Tuple[Any, str]]], predicted_data: List[List[Tuple[Any, str]]], otherTAG: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate NER performance at entity level across an entire corpus.
    
    Args:
        gold_data: List of sentences where each sentence is a list of (tupla, gold_tag) tuples
        predicted_data: List of sentences where each sentence is a list of (word, pred_tag) tuples
        
    Returns:
        Dictionary with overall precision, recall, and F1 scores
    """
    total_correct = 0
    total_gold = 0
    total_pred = 0
    all_confusion_matrix = {}
    all_entity_types = set()
    total_tokens = 0  # Track total tokens for O,O calculation
    
    for i in range(len(gold_data)):
        # Extract just the tags
        sentence = gold_data[i]
        sentence_pred = predicted_data[i]
        
        gold_tags = []
        pred_tags = []
        for j in range(len(sentence)):
            # Extract the tag of each word
            gold_tags.append(sentence[j][1])
            pred_tags.append(sentence_pred[j][1])
        
        total_tokens += len(gold_tags)  # Count total tokens
        
        # Extract entities
        gold_entities = extract_entities(gold_tags, otherTAG)
        pred_entities = extract_entities(pred_tags, otherTAG)
        
        # Track positions covered by entities in both gold and pred
        gold_positions = set()
        for _, start, end in gold_entities:
            for pos in range(start, end + 1):
                gold_positions.add(pos)
                
        pred_positions = set()
        for _, start, end in pred_entities:
            for pos in range(start, end + 1):
                pred_positions.add(pos)
        
        # Calculate O,O positions (true negatives)
        o_o_count = len(sentence) - len(gold_positions.union(pred_positions))
        if o_o_count > 0:
            all_confusion_matrix[("O", "O")] = all_confusion_matrix.get(("O", "O"), 0) + o_o_count
        
        # Evaluate this sentence
        results = evaluate_entities(gold_entities, pred_entities)
        sent_confusion_matrix = extract_confusion_matrix(gold_entities, pred_entities)
        sent_entity_types = extract_entities_types(gold_entities, pred_entities)
        
        # Update all entity types and confusion matrix
        all_entity_types.update(sent_entity_types)
        for key, value in sent_confusion_matrix.items():
            all_confusion_matrix[key] = all_confusion_matrix.get(key, 0) + value
    
        # Accumulate counts
        total_correct += results['correct']
        total_gold += results['gold_count']
        total_pred += results['pred_count']
    
    # Calculate overall metrics
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': all_confusion_matrix,
        'entity_types': all_entity_types
    }


# Extend CRFTagger to support entity-level evaluation
def entity_level_accuracy(tagger: CRFTagger, test_data: List[List[Tuple[Tuple[str, str, str], str]]], otherTAG: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate entity-level evaluation metrics for a CRFTagger.
    
    Args:
        tagger: Trained CRFTagger model
        test_data: List of sentences where each sentence is a list of ((word, pos, lema), tag) tuples
        
    Returns:
        Dictionary with precision, recall, F1, and accuracy scores
    """
    # Convert test data to the format expected by the evaluation function
    
    # Get predictions
    predicted_data = []
    for sentence in test_data:  # Use original test_data to extract words
        words = [tupla for tupla, _ in sentence]
        tags = tagger.tag(words)
        predicted_data.append(list(zip(words, tags)))
    
    # Evaluate
    results = evaluate_ner_corpus(test_data, predicted_data, otherTAG)

    # Return the data to construct the ConfusionMatrix
    
    
    return results



def plot_confusion_matrix(confusion_matrix: Dict[Tuple[str, str], int], entity_types: Set[str]) -> None:
    """Args:
        confusion_matrix: Dictionary with (gold_type, pred_type) keys and counts as values
        entity_types: Set of entity types to include in the matrix
    """

    # Make sure "O" is included for non-entity
    entity_types = sorted(list(entity_types) + ["O"])

    # Create a matrix filled with zeros
    matrix = np.zeros((len(entity_types), len(entity_types)))

    # Map entity types to indices
    type_to_idx = {t: i for i, t in enumerate(entity_types)}

    # Fill the matrix with confusion counts
    for (gold_type, pred_type), count in confusion_matrix.items():
        if gold_type in type_to_idx and pred_type in type_to_idx:
            gold_idx = type_to_idx[gold_type]
            pred_idx = type_to_idx[pred_type]
            matrix[gold_idx, pred_idx] = count

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    # Change fmt from 'd' to 'g' to handle both integers and floats
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", 
                xticklabels=entity_types, yticklabels=entity_types)
    plt.title('Entity-Level Confusion Matrix')
    plt.xlabel('Predicted Entity Type')
    plt.ylabel('True Entity Type')
    plt.tight_layout()
    plt.show()

    # Print analysis of most common confusions
    print("\nMost common entity type confusions:")
    confusion_counts = [(gold, pred, count) for (gold, pred), count in confusion_matrix.items()]
    confusion_counts.sort(key=lambda x: x[2], reverse=True)

    for gold, pred, count in confusion_counts[:5]:
        if gold != pred:
            print(f"  {gold} mistaken as {pred}: {count} times")
    

# Example of running a complete analysis with the optimal feature configuration
def run_optimal_configuration(model_path: Optional[str] = None, preprocessed_test: Optional[List[List[Tuple[Tuple[str, str, str], str]]]] = None, train_tags: Optional[List[List[Tuple[Tuple[str, str, str], str]]]] = None, otherTAG: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a complete analysis with the optimal feature configuration.
    
    Args:
        model_path: Optional path to a pre-trained model file. If provided, the model will be loaded
                   instead of being trained. For example: 'best_model_r4_B_C_C_L.crf.tagger'
    
    Returns:
        A tuple containing (tagger, evaluation_results)
    """
    # Create feature function with optimal settings (based on our findings)
    optimal_feat_func = OptimizedFeatFunc(
        use_Basic=True,
        use_context_words=True, 
        use_contex_POS_tag=True,
        use_lemas=True,  # Including lemmatization
        use_specific_caracteristics=False,
        use_EXTRA=False  # Including extra features
    )
    
    # Initialize CRF tagger with our feature function
    optimal_tagger = CRFTagger(feature_func=optimal_feat_func)

    if train_tags: # CASO CUANDO SE QUIERE ENTRENAR EL MODELO CON CODIFICACIONES DIFERENTES A BIO

        optimal_tagger.train(train_tags, 'TAGS{}_model.crf.tagger')
    
    print(f"Loading pre-trained model from {model_path}...")
        # Load the model instead of training
    if model_path:
        optimal_tagger.set_model_file(model_path)
    
    print("Evaluating model on test data...")
    # Evaluate using entity-level metrics
    if otherTAG:
        entity_results = entity_level_accuracy(optimal_tagger, preprocessed_test, otherTAG = otherTAG)
    else:
        entity_results = entity_level_accuracy(optimal_tagger, preprocessed_test)
    
    print("\n=== Entity-Level Evaluation Results ===")
    print(f"Precision: {entity_results['precision']:.4f}")
    print(f"Recall: {entity_results['recall']:.4f}")
    print(f"F1 Score: {entity_results['f1']:.4f}")

    # Create and display confusion matrix
    print("\n=== Entity-Level Confusion Matrix ===")
    plot_confusion_matrix(entity_results['confusion_matrix'], entity_results['entity_types'])
    
    return entity_results


def train_completo(processed_train: List[List[Tuple[Tuple[str, str, str], str]]], processed_val: List[List[Tuple[Tuple[str, str, str], str]]]) -> Tuple[List[List[str]], List[float], List[CRFTagger]]:
    # Define feature groups to test
    feature_groups = {
        "Basic": True,        # word, length, etc.
        "Context_Words": True,  # prev_word, next_word
        "Context_POS": True,    # POS tags of surrounding words
        "Specific": True,      # location_suffix, organization_precedent, etc.
        "Lemmas": True,        # Use lemmatization features
    }
    
    # Initialize variables to track best configurations
    best_features = []
    best_scores = []
    best_models = []
    
    print("Starting greedy feature selection...")
    
    # Iterate through feature counts (0 to n) INCLUDING ALSO THE BASELINE (NO FEATURES)
    for r in range(0, len(feature_groups) + 1):
        print(f"\n--- Finding best configuration with {r} features ---")
        
        # In first round, test all individual features
        # In subsequent rounds, only test combinations that include previous best features
        if r == 0:
            # Test no features (baseline)
            candidates = [[]]
        elif r == 1:
            # Test each feature individually
            candidates = [[feat] for feat in feature_groups.keys()]
        else:
            # Keep best features from previous round and test adding one more
            candidates = [best_features[-1] + [feat] for feat in feature_groups.keys() 
                         if feat not in best_features[-1]]
        
        best_config = None
        best_score = 0
        best_model = None
        
        # Test all candidate configurations for this round
        for candidate_features in candidates:
            # Create configuration with only these features enabled
            config = {feat: (feat in candidate_features) for feat in feature_groups.keys()}
            print(f"Testing configuration: {config} out of {len(candidates)}")
            
            # Test this configuration
            start_time = time.time()
            tagger, entity_metrics = evaluate_feature_combination(config, processed_train, processed_val)
            elapsed = time.time() - start_time
            
            # Check if this is the best so far
            f1_score = entity_metrics['f1']
            print(f"F1 Score: {f1_score:.4f}, Time: {elapsed:.2f} seconds")
            
            if f1_score > best_score:
                best_score = f1_score
                best_config = candidate_features
                best_model = tagger
        
        # Save best configuration for this round
        best_features.append(best_config)
        best_scores.append(best_score)
        best_models.append(best_model)
        
        # Save the best model for this round
        if best_model:
            # Generate a descriptive model name with round number and features
            features_str = '_'.join([k[:1] for k, v in {feat: True for feat in best_config}.items()])
            model_path = f'best_model_r{r}_{features_str}.crf.tagger'
            
            # Train and save the best model for this round
            best_model.train(processed_train, model_path)
            print(f"Saved best model with {r} features to {model_path}")
        
        print(f"Best configuration with {r} features: {best_config}")
        print(f"Best F1 score: {best_score:.4f}")
    
    # Print summary of all best configurations
    print("\n=== Summary of Greedy Feature Selection ===")
    for r in range(len(best_features)):
        print(f"Round {r+1}: Best features = {best_features[r]}, F1 score = {best_scores[r]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_scores)+1), best_scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Number of Features (Greedy Selection)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return best_features, best_scores, best_models

def evaluate_feature_combination(config: Dict[str, bool], processed_train: List[List[Tuple[Tuple[str, str, str], str]]], processed_val: List[List[Tuple[Tuple[str, str, str], str]]]) -> Tuple[CRFTagger, Dict[str, Any]]:
    """Evaluate a single feature configuration and return metrics"""
    # Create feature function with the specified configuration
    feat_func = OptimizedFeatFunc(
        use_Basic=config['Basic'], 
        use_context_words=config['Context_Words'], 
        use_contex_POS_tag=config['Context_POS'], 
        use_specific_caracteristics=config['Specific'],
        use_lemas=config['Lemmas']
    )
    
    # Create the CRF tagger with the feature function
    ct = CRFTagger(feature_func=feat_func)
    
    # Create a temporary model file for evaluation
    ct.train(processed_train, 'temp_model.crf.tagger')
    
    # Calculate entity-level metrics
    entity_metrics = entity_level_accuracy(ct, processed_val)
    
    return ct, entity_metrics


def bio_to_io(tagged_sent: List[List[Tuple[str, str, str]]]) -> List[List[Tuple[str, str, str]]]:
    """Convert BIO tagging to IO tagging
    
    Args:
        tagged_sent: List of sentences where each sentence is a list of tuples (word, pos, tag) in BIO format.
        
    Returns:
        List of sentences where each sentence is a list of tuples (word, pos, tag) in IO format.
    """
    io_data = []
    for sentence in tagged_sent:
        io_sentence = []
        for word, pos, tag in sentence:
            if tag.startswith('B-'):
                # Replace B- with I- for any entity tag
                entity_type = tag[2:]
                io_sentence.append((word, pos, f"I-{entity_type}"))
            else:
                # Keep I- tags and O tags as they are
                io_sentence.append((word, pos, tag))
        io_data.append(io_sentence)
    return io_data

def bio_to_bioe(tagged_sent: List[List[Tuple[str, str, str]]]) -> List[List[Tuple[str, str, str]]]:
    """Convert BIO tagging to BIOE tagging (BIOE incluye E to indicate the end of an entity)
    Args:
        tagged_sent: List of sentences where each sentence is a list of tuples (word, pos, tag) in BIO format.
        
    Returns:
        List of sentences where each sentence is a list of tuples (word, pos, tag) in BIOE format.
    """
    bioe_data = []
    for sentence in tagged_sent:
        bioe_sentence = []
        n = len(sentence)
        i = 0
        while i < n:
            word, pos, tag = sentence[i]
            
            if tag.startswith('B-'):
                entity_type = tag[2:]
                
                # Find the end of the entity
                j = i + 1
                is_multi_token = False
                while j < n and sentence[j][2] == f"I-{entity_type}":
                    is_multi_token = True
                    j += 1
                
                # Add the B- tag
                bioe_sentence.append((word, pos, tag))
                
                # Process all intermediate I- tags
                for k in range(i + 1, j - 1):
                    bioe_sentence.append(sentence[k])
                
                # Convert the last I- tag to E- if this is a multi-token entity
                if is_multi_token:
                    last_word, last_pos, last_tag = sentence[j - 1]
                    bioe_sentence.append((last_word, last_pos, f"E-{entity_type}"))
                    i = j
                else:
                    i += 1
            else:
                bioe_sentence.append((word, pos, tag))
                i += 1
                
        bioe_data.append(bioe_sentence)
    return bioe_data

def bio_to_bioes(tagged_sent: List[List[Tuple[str, str, str]]]) -> List[List[Tuple[str, str, str]]]:
    """Convert BIO tagging to BIOES tagging
    Args:
        tagged_sent: List of sentences where each sentence is a list of tuples (word, pos, tag) in BIO format.
        
    Returns:
        List of sentences where each sentence is a list of tuples (word, pos, tag) in BIOES format.
    """
    bioes_data = []
    for sentence in tagged_sent:
        bioes_sentence = []
        n = len(sentence)
        i = 0
        while i < n:
            word, pos, tag = sentence[i]
            
            if tag.startswith('B-'):
                entity_type = tag[2:]
                
                # Check if it's a singleton entity (no following I- tags)
                if i + 1 == n or not sentence[i+1][2].startswith(f"I-{entity_type}"):
                    bioes_sentence.append((word, pos, f"S-{entity_type}"))
                    i += 1
                else:
                    # It's the beginning of a multi-token entity
                    bioes_sentence.append((word, pos, tag))
                    i += 1
                    
                    # Process all the intermediate I- tags
                    while i < n and sentence[i][2] == f"I-{entity_type}":
                        # Check if this is the last I- tag
                        if i + 1 == n or sentence[i+1][2] != f"I-{entity_type}":
                            # Change last I- tag to E- tag
                            word_i, pos_i, _ = sentence[i]
                            bioes_sentence.append((word_i, pos_i, f"E-{entity_type}"))
                        else:
                            bioes_sentence.append(sentence[i])
                        i += 1
            else:
                bioes_sentence.append((word, pos, tag))
                i += 1
                
        bioes_data.append(bioes_sentence)
    return bioes_data

def bio_to_biow(tagged_sent: List[List[Tuple[str, str, str]]]) -> List[List[Tuple[str, str, str]]]:
    """Convert BIO tagging to BIOW tagging (BIOW incluye W to indicate single-token entities)
    Args:
        tagged_sent: List of sentences where each sentence is a list of tuples (word, pos, tag) in BIO format.
        
    Returns:
        List of sentences where each sentence is a list of tuples (word, pos, tag) in BIOW format.
    """
    biow_data = []
    for sentence in tagged_sent:
        biow_sentence = []
        n = len(sentence)
        
        for i, (word, pos, tag) in enumerate(sentence):
            if tag.startswith('B-'):
                entity_type = tag[2:]
                # Check if it's a single-token entity
                if (i + 1 == n) or (not sentence[i+1][2].startswith(f"I-{entity_type}")):
                    biow_sentence.append((word, pos, f"W-{entity_type}"))
                else:
                    biow_sentence.append((word, pos, tag))
            else:
                biow_sentence.append((word, pos, tag))
                
        biow_data.append(biow_sentence)
    return biow_data


## hay que guardar los modelos de tags y las salidas.
def test_with_othersCodes(train: List[List[Tuple[str, str, str]]], preprocessed_test: List[List[Tuple[Tuple[str, str, str], str]]], best_model_path: str) -> Dict[str, Dict[str, float]]:
    Codes = ['BIO', 'IO', 'BIOE', 'BIOES', 'BIOW']
    test_results = {}
    
    '''# Extract entities from test data (for comparison)
    test_entities = {}
    for i in range(len(preprocessed_test)):
        sentence = preprocessed_test[i]
        test_tags = []
        
        for j in range(len(sentence)):
            # Extract the tag from each word
            test_tags.append(sentence[j][1])
            
        # Store entities from original test data
        test_entities[i] = extract_entities(test_tags)'''

    # Test each tagging scheme
    for scheme in Codes[1:]:
        print(f"\nProcessing {scheme} tagging scheme...")
        
        # Convert training data to the current scheme
        if scheme == 'IO':
            train_tag_data = bio_to_io(train)
        elif scheme == 'BIOE':
            train_tag_data = bio_to_bioe(train)
        elif scheme == 'BIOES':
            train_tag_data = bio_to_bioes(train)
        elif scheme == 'BIOW':
            train_tag_data = bio_to_biow(train)
            
        # Process the converted training data for CRF
        preprocessed_train_tags = prepare_data_for_crf(train_tag_data, True)
        
        # Train a new model with this tagging scheme
        print(f"Training model with {scheme} tagging scheme...")
        scheme_model_path = f'{scheme}_model.crf.tagger'
        
        results = run_optimal_configuration(model_path=best_model_path, 
                                            preprocessed_test=preprocessed_test, 
                                            train_tags=preprocessed_train_tags, 
                                            otherTAG = scheme)
        # Store results
        test_results[scheme] = results
        
        # Print results
        print(f"\nResults for {scheme} tagging scheme:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        
    # Compare results
    print("\n=== Comparison of Tagging Schemes ===")
    schemes = ['BIO'] + Codes[1:]
    metrics = ['precision', 'recall', 'f1']
    
    # Add BIO results (default)
    bio_results = run_optimal_configuration(best_model_path, preprocessed_test)
    test_results['BIO'] = bio_results
    
    # Create DataFrame for plotting
    results_df = pd.DataFrame({
        scheme: {metric: test_results[scheme][metric] for metric in metrics}
        for scheme in schemes
    })
    
    # Plot comparison
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Performance Comparison of Different Tagging Schemes')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return test_results