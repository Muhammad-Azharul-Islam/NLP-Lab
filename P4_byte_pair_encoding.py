from collections import defaultdict

def get_stats(vocab):
    """
    Count frequency of character pairs in the vocabulary
    
    Args:
        vocab (dict): Dictionary mapping tokens to frequency
        
    Returns:
        defaultdict: Frequencies of adjacent symbol pairs
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Merge the most frequent pair in the vocabulary
    
    Args:
        pair (tuple): The pair of symbols to merge
        v_in (dict): Dictionary mapping tokens to frequency
        
    Returns:
        dict: Updated vocabulary with merged pairs
    """
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    
    for word, freq in v_in.items():
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = freq
    
    return v_out

def byte_pair_encoding(corpus, num_merges):
    """
    Apply the Byte Pair Encoding algorithm to a corpus of text
    
    Args:
        corpus (list): List of words in the corpus
        num_merges (int): Number of merge operations to perform
        
    Returns:
        tuple: (dictionary of words with their frequencies, 
                list of merge operations)
    """
    # Initialize vocabulary with character-level representation
    vocab = {}
    for word in corpus:
        # Add space between characters
        chars = ' '.join(list(word))
        if chars not in vocab:
            vocab[chars] = 0
        vocab[chars] += 1
    
    # Store merge operations
    merges = []
    
    # Perform merges
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
            
        # Get the most frequent pair
        best = max(pairs, key=pairs.get)
        merges.append(best)
        
        # Merge the pair in the vocabulary
        vocab = merge_vocab(best, vocab)
        
        print(f"Merge #{i+1}: {best} -> {''.join(best)}")
    
    return vocab, merges

def apply_bpe(word, merges):
    """
    Apply learned BPE merges to a new word
    
    Args:
        word (str): Word to tokenize
        merges (list): List of merge operations (tuples)
        
    Returns:
        str: Tokenized word with subword units separated by spaces
    """
    # Split word into individual characters
    chars = ' '.join(list(word))
    
    # Apply merges in the same order they were learned
    for pair in merges:
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        chars = chars.replace(bigram, replacement)
    
    return chars

if __name__ == "__main__":
    # Sample corpus
    corpus = ["low", "lowest", "newer", "wider", "learning"]
    
    # Number of merges to perform
    num_merges = 20
    
    print("Original corpus:", corpus)
    print(f"Performing {num_merges} BPE merges...")
    
    # Learn BPE merge operations
    vocab, merges = byte_pair_encoding(corpus, num_merges)
    
    print("\nFinal vocabulary:")
    for token, freq in vocab.items():
        print(f"  {token} ({freq})")
    
    print("\nApplying BPE to new words:")
    test_words = ["lowering", "widener", "learner"]
    
    for word in test_words:
        tokenized = apply_bpe(word, merges)
        print(f"  {word} -> {tokenized}") 