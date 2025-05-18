import string
class OptimizedFeatFunc:
    def __init__(self, use_Basic: bool = True, use_context_words: bool = True, use_contex_POS_tag: bool = True, use_specific_caracteristics: bool = True, use_lemas: bool = True):
        """
        Constructor de la clase de las funciones de características para el CRFTagger.
        Uso:
        - use_Basic: Si se deben usar características básicas (longitud, mayúsculas, etc.)
        - use_context_words: Si se deben usar palabras de contexto (palabra anterior y siguiente)
        - use_contex_POS_tag: Si se deben usar etiquetas POS de contexto (etiqueta anterior y siguiente)
        - use_specific_caracteristics: Si se deben usar características específicas (Gazetteer)
        - use_lemas: Si se deben usar lemas (forma base de la palabra)
       
        """
        self.use_basic_features = use_Basic
        self.use_context = use_context_words
        self.use_conext_POS_tags = use_contex_POS_tag
        self.use_specific_caracteristics = use_specific_caracteristics
        self.use_lema = use_lemas
        
    
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
        

    def __call__(self, tokens: list, idx: int) -> dict:
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

        # Guardar en caché
        self.cache[cache_key] = feats
        return feats
    
from typing import List, Tuple
import spacy
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

def extract_BIO(tags):
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

def extract_entities_types(gold_entities, pred_entities):
    entity_types = set()
    # Add entity types to our set
    for entity_type, _, _ in gold_entities:
        entity_types.add(entity_type)
    for entity_type, _, _ in pred_entities:
        entity_types.add(entity_type)
    return entity_types  # Added return statement

def extract_confusion_matrix(gold_entities, pred_entities):
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
    # For each gold entity, find if it was correctly predicted
    for gold_entity in gold_entities:
        gold_type, start, end = gold_entity
        
        # Look for predicted entity at the same position
        matched = False
        for pred_entity in pred_entities:
            pred_type, p_start, p_end = pred_entity

            # Check for false positives (predictions without gold)
            # Check if this prediction corresponds to any gold entity
            has_gold = any(g_start == p_start and g_end == p_end for _, g_start, g_end in gold_entities)
            
            if not has_gold:
                confusion_key = ("O", pred_type)  # "O" represents no gold entity
                confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1

            elif start == p_start and end == p_end:  # Comprobar 
                # Update confusion matrix
                confusion_key = (gold_type, pred_type)
                confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
                matched = True
                break
        
        # If no matching prediction was found, it's a false negative
        if not matched:
            confusion_key = (gold_type, "O")  # "O" represents no prediction
            confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
                
    return confusion_matrix

def extract_entities(tags, otherTAG = None):
    """
    Extract entity spans from a sequence of BIO tags.
    
    Args:
        tags: List of BIO tags (e.g., 'B-PER', 'I-PER', 'O')
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """

    if not otherTAG:
        entities = extract_BIO(tags)

    return entities
    


def evaluate_entities(gold_entities, pred_entities):
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

def evaluate_ner_corpus(gold_data, predicted_data, otherTAG = None):
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
    
    for i in range(len(gold_data)):
        # Extract just the tags
        sentence = gold_data[i]
        sentence_pred = predicted_data[i]
        
        gold_tags = []
        pred_tags = []
        for j in range(len(sentence)):
           # Extrae el tag de cada palabra.
            gold_tags.append(sentence[j][1])
            pred_tags.append(sentence_pred[j][1])
           
        # Extract entities
        gold_entities = extract_entities(gold_tags, otherTAG)
        pred_entities = extract_entities(pred_tags, otherTAG)
        
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
    precision = total_correct / total_pred if total_pred > 0 else 0.0 # total_pred incluye las predicciones correctas y las incorrectas y además las predichas "inventadas".
    recall = total_correct / total_gold if total_gold > 0 else 1.0 # total_gold incluye las correctas predichas y las incorrectas y además las que no se han predicho.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': all_confusion_matrix,
        'entity_types': all_entity_types
    }


# Extend CRFTagger to support entity-level evaluation
def entity_level_accuracy(tagger, test_data, otherTAG = None):
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

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix: Dict[Tuple[str, str], int], entity_types: set) -> None:
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
def run_optimal_configuration(model_path=None, preprocessed_test = None, train_tags = None, otherTAG = None):
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
        use_contex_POS_tag=False,
        use_lemas=True,  # Including lemmatization
        use_specific_caracteristics=True
    )
    
    # Initialize CRF tagger with our feature function
    optimal_tagger = CRFTagger(feature_func=optimal_feat_func)

    if train_tags: # CASO CUANDO SE QUIERE ENTRENAR EL MODELO CON CODIFICACIONES DIFERENTES A BIO

        optimal_tagger.train(train_tags, 'TAGS{}_model.crf.tagger')
    
    print(f"Loading pre-trained model from {model_path}...")
        # Load the model instead of training
    optimal_tagger.set_model_file(model_path)
    
    print("Evaluating model on test data...")
    # Evaluate using entity-level metrics
    if train_tags:
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