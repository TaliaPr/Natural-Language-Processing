def entity_level_accuracy_debug(tagger, test_data, debug=False):
    """
    Calculate entity-level evaluation metrics for a CRFTagger with optional debugging.
    
    Args:
        tagger: Trained CRFTagger model
        test_data: List of sentences where each sentence is a list of (word, pos, tag) tuples
        debug: Whether to print debug information
        
    Returns:
        Dictionary with precision, recall, F1, and accuracy scores
    """
    if debug:
        print(f"Processing {len(test_data)} test sentences")
    
    # Convert test data to the format expected by the evaluation function
    formatted_test_data = [[(word, label) for (word, pos, label) in sent] for sent in test_data]
    
    if debug:
        print(f"Sample formatted test data: {formatted_test_data[0][:3] if formatted_test_data else []}")
    
    # Get predictions
    predicted_data = []
    for sentence in test_data:
        try:
            # Extract only words from the original format (word, pos, label)
            words = [word for word, _, _ in sentence]
            
            # Get tags from model
            tags = tagger.tag(words)
            
            # Combine into (word, tag) format
            predicted_data.append(list(zip(words, tags)))
            
        except Exception as e:
            if debug:
                print(f"Error processing sentence: {e}")
                print(f"Problem sentence: {sentence[:5]}...")
            continue
    
    if debug:
        print(f"Processed {len(predicted_data)} sentences for prediction")
        print(f"Sample prediction: {predicted_data[0][:3] if predicted_data else []}")
    
    # Evaluate
    # Make sure both lists are the same length
    min_len = min(len(formatted_test_data), len(predicted_data))
    if min_len < len(formatted_test_data) or min_len < len(predicted_data):
        if debug:
            print(f"WARNING: Mismatched lengths. Formatted: {len(formatted_test_data)}, Predicted: {len(predicted_data)}")
            print(f"Using only the first {min_len} sentences for evaluation")
    
    results = evaluate_ner_corpus_debug(
        formatted_test_data[:min_len], 
        predicted_data[:min_len],
        debug=debug
    )
    
    return results

def evaluate_ner_corpus_debug(gold_data, predicted_data, debug=False):
    """
    Evaluate NER performance at entity level across an entire corpus with debugging.
    
    Args:
        gold_data: List of sentences where each sentence is a list of (word, gold_tag) tuples
        predicted_data: List of sentences where each sentence is a list of (word, pred_tag) tuples
        debug: Whether to print debug information
        
    Returns:
        Dictionary with overall precision, recall, and F1 scores
    """
    total_correct = 0
    total_gold = 0
    total_pred = 0
    
    if debug:
        print(f"Evaluating {len(gold_data)} sentence pairs")
    
    for i, (gold_sent, pred_sent) in enumerate(zip(gold_data, predicted_data)):
        try:
            # Extract just the tags
            gold_tags = [tag for _, tag in gold_sent]
            pred_tags = [tag for _, tag in pred_sent]
            
            if debug and i < 3:  # Debug output for the first few sentences
                print(f"\nSentence {i+1}:")
                print(f"Gold tags: {gold_tags[:10]}{'...' if len(gold_tags) > 10 else ''}")
                print(f"Pred tags: {pred_tags[:10]}{'...' if len(pred_tags) > 10 else ''}")
            
            # Extract entities
            gold_entities = extract_entities_debug(gold_tags, debug=(debug and i < 3))
            pred_entities = extract_entities_debug(pred_tags, debug=(debug and i < 3))
            
            if debug and i < 3:
                print(f"Gold entities: {gold_entities}")
                print(f"Predicted entities: {pred_entities}")
            
            # Evaluate this sentence
            correct = len(set(gold_entities) & set(pred_entities))
            
            if debug and i < 3:
                print(f"Correct: {correct} / Gold: {len(gold_entities)} / Pred: {len(pred_entities)}")
            
            # Accumulate counts
            total_correct += correct
            total_gold += len(gold_entities)
            total_pred += len(pred_entities)
        
        except Exception as e:
            if debug:
                print(f"Error evaluating sentence {i}: {e}")
            continue
    
    # Calculate overall metrics
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Summary statistics
    if debug:
        print(f"\nOverall statistics:")
        print(f"Total correct entities: {total_correct}")
        print(f"Total gold entities: {total_gold}")
        print(f"Total predicted entities: {total_pred}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': f1  # Using F1 as the "accuracy" metric for entity-level evaluation
    }

def extract_entities_debug(tags, debug=False):
    """
    Extract entity spans from a sequence of BIO tags with debugging.
    
    Args:
        tags: List of BIO tags (e.g., 'B-PER', 'I-PER', 'O')
        debug: Whether to print debug information
        
    Returns:
        List of tuples (entity_type, start_idx, end_idx)
    """
    entities = []
    entity_type = None
    start_idx = None
    
    if debug:
        print(f"Extracting entities from {len(tags)} tags")
    
    for i, tag in enumerate(tags):
        try:
            # Handle different tag formats
            if tag == 'O':
                # End any current entity
                if entity_type is not None:
                    entities.append((entity_type, start_idx, i - 1))
                    if debug:
                        print(f"Ending entity at O: {entity_type}, {start_idx}-{i-1}")
                    entity_type = None
                    start_idx = None
            
            elif tag.startswith('B-'):
                # End any current entity
                if entity_type is not None:
                    entities.append((entity_type, start_idx, i - 1))
                    if debug:
                        print(f"Ending entity at B-: {entity_type}, {start_idx}-{i-1}")
                
                # Start a new entity
                entity_type = tag[2:]  # Remove 'B-' prefix
                start_idx = i
                if debug:
                    print(f"Starting new entity: {entity_type} at position {i}")
            
            elif tag.startswith('I-'):
                curr_type = tag[2:]  # Remove 'I-' prefix
                
                # Handle various I- cases
                if entity_type is None:
                    # I- without preceding B-, start a new entity
                    entity_type = curr_type
                    start_idx = i
                    if debug:
                        print(f"Starting entity from I- without B-: {entity_type} at position {i}")
                
                elif curr_type != entity_type:
                    # I- with different type, end current and start new
                    entities.append((entity_type, start_idx, i - 1))
                    if debug:
                        print(f"Ending entity at mismatched I-: {entity_type}, {start_idx}-{i-1}")
                    
                    entity_type = curr_type
                    start_idx = i
                    if debug:
                        print(f"Starting new entity from mismatched I-: {entity_type} at position {i}")
                
                # If same type, continue current entity (nothing to do)
            
            else:
                # Unknown tag format
                if debug:
                    print(f"Unknown tag format at position {i}: {tag}")
                
                # End any current entity
                if entity_type is not None:
                    entities.append((entity_type, start_idx, i - 1))
                    if debug:
                        print(f"Ending entity at unknown tag: {entity_type}, {start_idx}-{i-1}")
                    entity_type = None
                    start_idx = None
        
        except Exception as e:
            if debug:
                print(f"Error processing tag at position {i}: {tag}, Error: {e}")
    
    # Don't forget the last entity if the sequence ends with an entity
    if entity_type is not None:
        entities.append((entity_type, start_idx, len(tags) - 1))
        if debug:
            print(f"Ending final entity: {entity_type}, {start_idx}-{len(tags)-1}")
    
    if debug:
        print(f"Extracted {len(entities)} entities: {entities}")
    
    return entities

# Usage:
# entity_metrics = entity_level_accuracy_debug(trained_tagger, test, debug=True)
