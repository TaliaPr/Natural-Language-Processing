class train_data:
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
    
    def extract_entities(self, tags=None):
        """
        Extract entity spans from a sequence of BIO tags.
        
        Args:
            tags: List of BIO tags (e.g., 'B-PER', 'I-PER', 'O'). If None, uses self._bio
            
        Returns:
            List of tuples (entity_type, start_idx, end_idx)
        """
        if tags is None:
            tags = self._bio
            
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
    
    def evaluate_entities(self, gold_entities, pred_entities):
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
        
        # Calculate precision, recall, and F1
        precision = correct / len(pred_set) if pred_set else 0.0
        recall = correct / len(gold_set) if gold_set else 1.0  # Perfect recall if no gold entities
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'gold_count': len(gold_set),
            'pred_count': len(pred_set),
            'correct': correct
        }
    
    def evaluate_against(self, predicted_data):
        """
        Evaluate NER performance at entity level against predicted data.
        
        Args:
            predicted_data: List of sentences where each sentence is a list of (word, pred_tag) tuples
            or another test_data object
            
        Returns:
            Dictionary with overall precision, recall, and F1 scores
        """
        # If the input is another test_data object, extract its word_bio tuples
        if isinstance(predicted_data, test_data):
            predicted_data = [predicted_data.get_word_bio()]
        # If it's a single sentence (list of tuples), wrap it in a list
        elif isinstance(predicted_data[0], tuple):
            predicted_data = [predicted_data]
            
        # Get our own word_bio tuples as a list of sentences
        gold_data = [self.get_word_bio()]
        
        total_correct = 0
        total_gold = 0
        total_pred = 0
        
        for gold_sent, pred_sent in zip(gold_data, predicted_data):
            # Extract just the tags
            gold_tags = [tag for _, tag in gold_sent]
            pred_tags = [tag for _, tag in pred_sent]
            
            # Extract entities
            gold_entities = self.extract_entities(gold_tags)
            pred_entities = self.extract_entities(pred_tags)
            
            # Evaluate this sentence
            results = self.evaluate_entities(gold_entities, pred_entities)
            
            # Accumulate counts
            total_correct += results['correct']
            total_gold += results['gold_count']
            total_pred += results['pred_count']
        
        # Calculate overall metrics
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_gold if total_gold > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_correct / total_gold if total_gold > 0 else 1.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy  # Using recall as the "accuracy" metric for entity-level evaluation
        }
    
    def entity_level_accuracy(self, tagger, test_data=None):
        """
        Calculate entity-level evaluation metrics for a CRFTagger.
        
        Args:
            tagger: Trained CRFTagger model
            test_data: List of sentences where each sentence is a list of (word, pos, tag) tuples
                       If None, uses this instance's data
            
        Returns:
            Dictionary with precision, recall, F1, and accuracy scores
        """
        # If no test data is provided, use this instance's data
        if test_data is None:
            # Convert current instance data to format needed for evaluation
            words = self._words
            tags = self._bio
            
            # Get predictions for current instance
            predicted_tags = tagger.tag(words)
            
            # Create gold and predicted data in the format needed for evaluation
            gold_data = [[(word, tag) for word, tag in zip(words, tags)]]
            predicted_data = [[(word, pred) for word, pred in zip(words, predicted_tags)]]
            
            # Evaluate
            return self.evaluate_corpus(gold_data, predicted_data)
        
        # Convert test data to the format expected by the evaluation function
        formatted_test_data = [[(word, label) for (word, pos, label) in sent] for sent in test_data]
        
        # Get predictions
        predicted_data = []
        for sentence in test_data:
            words = [word for word, _, _ in sentence]
            tags = tagger.tag(words)
            predicted_data.append(list(zip(words, tags)))
        
        # Evaluate
        return self.evaluate_against(formatted_test_data, predicted_data)
    