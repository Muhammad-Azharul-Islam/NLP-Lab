import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess text for Word2Vec training
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokenized sentences
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize sentences into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = []
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        # Filter out stopwords and tokens with less than 2 characters
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        if filtered_tokens:  # Only add non-empty sentences
            tokenized_sentences.append(filtered_tokens)
    
    return tokenized_sentences

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
    """
    Train a Word2Vec model on the given sentences
    
    Args:
        sentences (list): List of tokenized sentences
        vector_size (int): Dimensionality of the word vectors
        window (int): Maximum distance between the current and predicted word
        min_count (int): Minimum frequency of words to be considered
        workers (int): Number of worker threads
        epochs (int): Number of training epochs
        
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    # Train a Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    
    return model

def load_pretrained_word2vec():
    """
    Load pre-trained Word2Vec models
    
    Returns:
        KeyedVectors: Pre-trained word vectors
    """
    # Load pre-trained Word2Vec model (Google's Word2Vec trained on Google News)
    # Note: You need to download this file separately from:
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin', binary=True
        )
        return model
    except FileNotFoundError:
        print("Pre-trained model file not found. Please download it from the Google Drive link.")
        return None

def visualize_word_vectors(model, words, filename='word_vectors.png'):
    """
    Visualize word vectors using t-SNE
    
    Args:
        model: Word2Vec model
        words (list): List of words to visualize
        filename (str): Output file name
    """
    # Get word vectors
    word_vectors = []
    valid_words = []
    
    for word in words:
        try:
            vector = model.wv[word]
            word_vectors.append(vector)
            valid_words.append(word)
        except KeyError:
            continue
    
    if len(valid_words) < 2:
        print("Not enough valid words to visualize")
        return
    
    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    # Plot the word vectors
    plt.figure(figsize=(12, 8))
    
    # Plot points
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', alpha=0.5)
    
    # Add word labels
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', 
                     ha='right', va='bottom', fontsize=10)
    
    plt.title('t-SNE Visualization of Word Vectors')
    plt.savefig(filename)
    plt.close()
    
    print(f"Word vector visualization saved as '{filename}'")

def explore_word_relationships(model, word_pairs):
    """
    Explore relationships between word pairs using Word2Vec
    
    Args:
        model: Word2Vec model
        word_pairs (list): List of tuples containing word pairs
    """
    print("\nWord Similarities:")
    for word1, word2 in word_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        except KeyError:
            print(f"One or both words ('{word1}', '{word2}') not in vocabulary")
    
    print("\nMost similar words:")
    for word in [pair[0] for pair in word_pairs]:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\nMost similar to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
    
    print("\nWord analogies:")
    analogies = [
        ('king', 'man', 'woman'),  # king - man + woman = queen
        ('france', 'paris', 'berlin'),  # france - paris + berlin = germany
        ('good', 'better', 'bad')  # good - better + bad = worse
    ]
    
    for word1, word2, word3 in analogies:
        try:
            result = model.wv.most_similar(positive=[word3, word2], negative=[word1], topn=1)
            print(f"{word2} is to {word1} as {result[0][0]} is to {word3} ({result[0][1]:.4f})")
        except KeyError:
            print(f"One or more words in ('{word1}', '{word2}', '{word3}') not in vocabulary")

if __name__ == "__main__":
    # Sample text for training
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
    
    The field of NLP involves a wide range of tasks and approaches. Text classification, part-of-speech tagging, named
    entity recognition, sentiment analysis, and machine translation are all common NLP tasks. Modern NLP approaches
    often use neural networks and deep learning methods to achieve state-of-the-art results.
    """
    
    # Part A: Using a pre-trained Word2Vec model
    print("Part A: Using pre-trained Word2Vec model")
    
    # This is commented out because the file is large and needs to be downloaded separately
    print("Note: To use pre-trained Word2Vec from Google News, download the model file first")
    print("Simulating pre-trained model behavior...")
    
    # Part B: Training a Word2Vec model on custom corpus
    print("\nPart B: Training Word2Vec on custom corpus")
    
    # Preprocess the text
    sentences = preprocess_text(sample_text)
    print(f"Number of sentences for training: {len(sentences)}")
    
    # Train the Word2Vec model
    model = train_word2vec_model(sentences)
    print("Word2Vec model trained successfully")
    
    # Get vocabulary size
    vocab_size = len(model.wv.index_to_key)
    print(f"Vocabulary size: {vocab_size}")
    
    # Get some sample vectors
    sample_word = model.wv.index_to_key[0]
    print(f"\nExample word vector for '{sample_word}':")
    print(model.wv[sample_word][:10])  # Show first 10 dimensions
    
    # Define word pairs to explore
    word_pairs = [
        ('language', 'processing'),
        ('computer', 'machine'),
        ('natural', 'artificial'),
        ('intelligence', 'learning')
    ]
    
    # Explore word relationships
    explore_word_relationships(model, word_pairs)
    
    # Visualize word vectors
    words_to_visualize = []
    for pair in word_pairs:
        words_to_visualize.extend(pair)
    words_to_visualize.extend(['data', 'text', 'neural', 'network', 'speech'])
    
    visualize_word_vectors(model, words_to_visualize)
    
    # Save the model
    model.save("word2vec_sample.model")
    print("Model saved as 'word2vec_sample.model'") 