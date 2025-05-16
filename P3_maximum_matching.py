def max_match_tokenize(text, dictionary):
    """
    Tokenize text using the Maximum Matching algorithm.
    
    Args:
        text (str): The input text to tokenize
        dictionary (set): A set of valid words
        
    Returns:
        list: A list of tokenized words
    """
    if not text:
        return []
    
    # Set initial maximum word length to consider
    # (or use the longest word in the dictionary)
    max_word_length = max(len(word) for word in dictionary) if dictionary else 10
    
    tokens = []
    start_idx = 0
    
    while start_idx < len(text):
        # Find the longest match starting from current position
        found_match = False
        end_idx = min(start_idx + max_word_length, len(text))
        
        while end_idx > start_idx:
            word_candidate = text[start_idx:end_idx]
            if word_candidate in dictionary:
                # Found a match
                tokens.append(word_candidate)
                start_idx = end_idx
                found_match = True
                break
            end_idx -= 1
        
        if not found_match:
            # If no match found, take the first character as a token
            tokens.append(text[start_idx])
            start_idx += 1
    
    return tokens

def load_dictionary(file_path=None):
    """
    Load a dictionary from a file or create a sample dictionary
    
    Args:
        file_path (str, optional): Path to dictionary file
        
    Returns:
        set: A set of dictionary words
    """
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    else:
        # Sample dictionary for demonstration purposes
        return {
            "natural", "language", "processing", "is", "a", "field", "of", 
            "artificial", "intelligence", "that", "focuses", "on", "the", 
            "interaction", "between", "computers", "and", "humans", "using", 
            "natural", "human", "languages", "computer", "science", "machine", 
            "learning", "computational", "linguistics"
        }

if __name__ == "__main__":
    # Create or load a dictionary
    dictionary = load_dictionary()
    
    # Sample text to tokenize
    sample_text = "naturallanguageprocessingisafieldofartificialintelligence"
    
    # Apply forward maximum matching
    tokens = max_match_tokenize(sample_text, dictionary)
    
    # Display results
    print("Input text:", sample_text)
    print("Tokenized result:", tokens)
    print("Joined tokens:", " ".join(tokens))
    
    # Another example with spaces (will be treated as tokens if not in dictionary)
    sample_text2 = "natural language processing is cool"
    tokens2 = max_match_tokenize(sample_text2, dictionary)
    print("\nInput text:", sample_text2)
    print("Tokenized result:", tokens2)
    
    # Example of how the algorithm handles unknown words
    sample_text3 = "naturallanguageprocessingiscool"
    tokens3 = max_match_tokenize(sample_text3, dictionary)
    print("\nInput text with unknown word:", sample_text3)
    print("Tokenized result:", tokens3) 