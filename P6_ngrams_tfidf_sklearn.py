import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

def extract_ngrams_sklearn(documents, n):
    """
    Extract n-grams using sklearn's CountVectorizer
    
    Args:
        documents (list): List of document strings
        n (int): n-gram size (1 for unigrams, 2 for bigrams, etc.)
        
    Returns:
        tuple: (feature matrix, feature names)
    """
    # Initialize the vectorizer
    vectorizer = CountVectorizer(ngram_range=(n, n))
    
    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return X.toarray(), feature_names

def calculate_tfidf_sklearn(documents, ngram_range=(1, 1)):
    """
    Calculate TF-IDF using sklearn's TfidfVectorizer
    
    Args:
        documents (list): List of document strings
        ngram_range (tuple): Range of n-gram sizes to include
        
    Returns:
        tuple: (TF-IDF matrix, feature names)
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    
    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return X.toarray(), feature_names

def show_top_features(feature_matrix, feature_names, document_idx, top_n=10):
    """
    Show top features for a specific document
    
    Args:
        feature_matrix (numpy.ndarray): Feature matrix
        feature_names (numpy.ndarray): Array of feature names
        document_idx (int): Index of the document to analyze
        top_n (int): Number of top features to show
    """
    # Get the document vector
    document_vector = feature_matrix[document_idx]
    
    # Get the indices of the top features
    top_indices = np.argsort(document_vector)[::-1][:top_n]
    
    # Get the feature names and values
    top_features = [(feature_names[i], document_vector[i]) for i in top_indices]
    
    return top_features

if __name__ == "__main__":
    # Sample documents
    documents = [
        "Natural language processing is a field of artificial intelligence",
        "Machine learning algorithms are used in NLP",
        "Deep learning is a subset of machine learning",
        "NLP applications include translation and sentiment analysis"
    ]
    
    print("Sample Documents:")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc}")
    
    # Extract and display n-grams
    for n in range(1, 4):  # unigrams, bigrams, trigrams
        print(f"\n{n}-grams from all documents:")
        
        ngram_counts, ngram_features = extract_ngrams_sklearn(documents, n)
        
        print(f"Total {n}-gram features: {len(ngram_features)}")
        print(f"Sample {n}-grams: {', '.join(ngram_features[:10])}")
        
        # Display counts for the first document
        doc_idx = 0
        doc_ngrams = [(feature, ngram_counts[doc_idx, i]) 
                      for i, feature in enumerate(ngram_features) 
                      if ngram_counts[doc_idx, i] > 0]
        
        print(f"\n{n}-grams in Document 1:")
        for feature, count in doc_ngrams:
            print(f"  '{feature}': {count}")
    
    # Calculate TF-IDF with various n-gram ranges
    for ngram_range in [(1, 1), (1, 2), (1, 3)]:
        print(f"\nTF-IDF with n-gram range {ngram_range}:")
        
        tfidf_matrix, tfidf_features = calculate_tfidf_sklearn(documents, ngram_range)
        
        print(f"Total features: {len(tfidf_features)}")
        
        # Show top TF-IDF features for each document
        print("\nTop features by TF-IDF score:")
        for doc_idx in range(len(documents)):
            top_features = show_top_features(tfidf_matrix, tfidf_features, doc_idx, top_n=5)
            
            print(f"\nDocument {doc_idx + 1}:")
            for feature, score in top_features:
                print(f"  '{feature}': {score:.4f}") 