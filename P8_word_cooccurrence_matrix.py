import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
from scipy import spatial
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess text by removing special characters, converting to lowercase, 
    and removing stopwords
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return filtered_tokens

def build_vocabulary(tokens):
    """
    Build vocabulary from tokens
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        dict: Mapping from token to index
    """
    vocabulary = {}
    for i, token in enumerate(sorted(set(tokens))):
        vocabulary[token] = i
    
    return vocabulary

def create_cooccurrence_matrix(tokens, vocabulary, window_size=2):
    """
    Create co-occurrence matrix
    
    Args:
        tokens (list): List of tokens
        vocabulary (dict): Mapping from token to index
        window_size (int): Context window size
        
    Returns:
        numpy.ndarray: Co-occurrence matrix
    """
    vocab_size = len(vocabulary)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    
    for i, current_token in enumerate(tokens):
        # Get the index of the current token
        current_idx = vocabulary[current_token]
        
        # Define the context window
        context_start = max(0, i - window_size)
        context_end = min(len(tokens), i + window_size + 1)
        
        # Update co-occurrence counts
        for j in range(context_start, context_end):
            if i != j:  # Skip the current token
                context_token = tokens[j]
                context_idx = vocabulary[context_token]
                cooccurrence_matrix[current_idx, context_idx] += 1
    
    return cooccurrence_matrix

def calculate_cosine_similarity(matrix, word1, word2, vocabulary):
    """
    Calculate cosine similarity between word vectors
    
    Args:
        matrix (numpy.ndarray): Co-occurrence matrix
        word1 (str): First word
        word2 (str): Second word
        vocabulary (dict): Mapping from word to index
        
    Returns:
        float: Cosine similarity
    """
    if word1 not in vocabulary or word2 not in vocabulary:
        return None
    
    idx1 = vocabulary[word1]
    idx2 = vocabulary[word2]
    
    vec1 = matrix[idx1]
    vec2 = matrix[idx2]
    
    # Calculate cosine similarity
    similarity = 1 - spatial.distance.cosine(vec1, vec2)
    
    return similarity

def find_most_similar_words(matrix, word, vocabulary, top_n=5):
    """
    Find most similar words to a given word
    
    Args:
        matrix (numpy.ndarray): Co-occurrence matrix
        word (str): Target word
        vocabulary (dict): Mapping from word to index
        top_n (int): Number of top similar words to return
        
    Returns:
        list: List of tuples (word, similarity score)
    """
    if word not in vocabulary:
        return []
    
    idx = vocabulary[word]
    word_vector = matrix[idx]
    
    # Calculate similarity with all other words
    similarities = []
    inv_vocabulary = {idx: word for word, idx in vocabulary.items()}
    
    for i in range(len(vocabulary)):
        if i != idx:  # Skip the word itself
            other_vector = matrix[i]
            similarity = 1 - spatial.distance.cosine(word_vector, other_vector)
            similarities.append((inv_vocabulary[i], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

def plot_cooccurrence_heatmap(matrix, vocabulary, top_n=20):
    """
    Plot a heatmap of the co-occurrence matrix
    
    Args:
        matrix (numpy.ndarray): Co-occurrence matrix
        vocabulary (dict): Mapping from word to index
        top_n (int): Number of top frequent words to include
    """
    # Get word frequencies (row sums)
    word_freq = np.sum(matrix, axis=1)
    
    # Get top N frequent words
    inv_vocabulary = {idx: word for word, idx in vocabulary.items()}
    top_indices = np.argsort(word_freq)[-top_n:]
    
    # Extract submatrix and word labels
    submatrix = matrix[top_indices][:, top_indices]
    word_labels = [inv_vocabulary[idx] for idx in top_indices]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(submatrix, index=word_labels, columns=word_labels)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(df, cmap='hot')
    plt.colorbar(label='Co-occurrence count')
    
    # Add labels
    plt.xticks(range(len(word_labels)), word_labels, rotation=45, ha='right')
    plt.yticks(range(len(word_labels)), word_labels)
    
    plt.title(f'Co-occurrence Matrix for Top {top_n} Words')
    plt.tight_layout()
    plt.savefig('cooccurrence_heatmap.png')
    plt.close()
    
    print(f"Heatmap saved as 'cooccurrence_heatmap.png'")

if __name__ == "__main__":
    # Sample text
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
    It is concerned with the interactions between computers and human language, in particular how to program computers
    to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the
    contents of documents, including the contextual nuances of the language within them. The technology can then
    accurately extract information and insights contained in the documents as well as categorize and organize the
    documents themselves.
    
    Challenges in natural language processing frequently involve speech recognition, natural language understanding,
    and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing
    published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test
    as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language,
    but at the time not articulated as a problem separate from artificial intelligence.
    """
    
    # Preprocess text
    tokens = preprocess_text(sample_text)
    
    # Build vocabulary
    vocabulary = build_vocabulary(tokens)
    
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Sample tokens: {tokens[:10]}")
    
    # Create co-occurrence matrix with different window sizes
    for window_size in [1, 2, 5]:
        print(f"\nCreating co-occurrence matrix with window size {window_size}...")
        cooccurrence_matrix = create_cooccurrence_matrix(tokens, vocabulary, window_size)
        
        print(f"Co-occurrence matrix shape: {cooccurrence_matrix.shape}")
        
        # Print example pairs and their co-occurrence counts
        print("\nExample co-occurrence counts:")
        if "language" in vocabulary and "processing" in vocabulary:
            idx1 = vocabulary["language"]
            idx2 = vocabulary["processing"]
            print(f"'language' and 'processing': {cooccurrence_matrix[idx1, idx2]}")
        
        if "natural" in vocabulary and "language" in vocabulary:
            idx1 = vocabulary["natural"]
            idx2 = vocabulary["language"]
            print(f"'natural' and 'language': {cooccurrence_matrix[idx1, idx2]}")
        
        # Calculate similarity between words
        print("\nWord similarities:")
        word_pairs = [
            ("language", "processing"),
            ("natural", "language"),
            ("computer", "artificial"),
            ("natural", "artificial")
        ]
        
        for word1, word2 in word_pairs:
            similarity = calculate_cosine_similarity(cooccurrence_matrix, word1, word2, vocabulary)
            if similarity is not None:
                print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        
        # Find most similar words to a target word
        target_word = "language"
        if target_word in vocabulary:
            print(f"\nMost similar words to '{target_word}':")
            similar_words = find_most_similar_words(cooccurrence_matrix, target_word, vocabulary)
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
    
    # Plot heatmap for the last computed matrix
    plot_cooccurrence_heatmap(cooccurrence_matrix, vocabulary) 