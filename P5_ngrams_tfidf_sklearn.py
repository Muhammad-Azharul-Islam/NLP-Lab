import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def extract_ngrams_sklearn(documents, n):
    # Initialize the vectorizer
    vectorizer = CountVectorizer(ngram_range=(n, n))
    
    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return X.toarray(), feature_names

def calculate_tfidf_sklearn(documents, ngram_range=(1, 1)):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    
    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return X.toarray(), feature_names

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
    
    # Generate and display n-grams for each document
    for n in range(1, 4):  # unigrams, bigrams, trigrams
        print(f"\n{n}-grams for all documents:")
        
        # Extract n-grams using sklearn
        ngram_counts, ngram_features = extract_ngrams_sklearn(documents, n)
        
        # Display n-grams for each document
        for doc_idx in range(len(documents)):
            print(f"\nDocument {doc_idx+1} {n}-grams:")
            
            # Get n-grams that appear in this document
            doc_ngrams = [(feature, ngram_counts[doc_idx, i]) 
                          for i, feature in enumerate(ngram_features) 
                          if ngram_counts[doc_idx, i] > 0]
            
            # Display all n-grams for this document
            for i, (ngram, count) in enumerate(doc_ngrams):
                print(f"  {i+1}. {ngram}")
    
    # Calculate TF-IDF for unigrams
    print("\nTF-IDF Scores for all documents:")
    tfidf_matrix, tfidf_features = calculate_tfidf_sklearn(documents)
    
    # Display TF-IDF scores for each document
    for doc_idx in range(len(documents)):
        print(f"\nDocument {doc_idx+1}:")
        
        # Get TF-IDF scores for this document
        doc_tfidf = [(feature, tfidf_matrix[doc_idx, i]) 
                     for i, feature in enumerate(tfidf_features) 
                     if tfidf_matrix[doc_idx, i] > 0]
        
        # Sort by TF-IDF score (descending)
        doc_tfidf.sort(key=lambda x: x[1], reverse=True)
        
        # Display all TF-IDF scores
        for term, score in doc_tfidf:
            print(f"  {term}: {score:.4f}") 