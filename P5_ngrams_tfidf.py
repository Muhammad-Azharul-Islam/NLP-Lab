import re
import math
import numpy as np
from collections import Counter, defaultdict

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text

def generate_ngrams(text, n):
    # Tokenize text into words
    words = text.split()
    
    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
        
    return ngrams

def calculate_tf(document):
    # Count term frequencies
    tf_dict = Counter(document)
    
    # Normalize by document length
    doc_length = len(document)
    for term in tf_dict:
        tf_dict[term] = tf_dict[term] / doc_length
        
    return dict(tf_dict)

def calculate_idf(documents):
    # Count the number of documents containing each term
    term_doc_count = defaultdict(int)
    for doc in documents:
        # Get unique terms in the document
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_count[term] += 1
    
    # Calculate IDF for each term using sklearn's formula:
    # log((1 + total_docs) / (1 + doc_count)) + 1
    idf_dict = {}
    total_docs = len(documents)
    
    for term, doc_count in term_doc_count.items():
        # sklearn formula with smooth_idf=True (default)
        idf_dict[term] = math.log((1 + total_docs) / (1 + doc_count)) + 1
        
    return idf_dict

def calculate_tfidf(tf_dict, idf_dict):
    tfidf_dict = {}
    
    for term, tf in tf_dict.items():
        if term in idf_dict:
            tfidf_dict[term] = tf * idf_dict[term]
    
    # Apply L2 normalization (Euclidean norm) to match sklearn
    values = np.array(list(tfidf_dict.values()))
    norm = np.sqrt(np.sum(values ** 2))
    
    # Avoid division by zero
    if norm > 0:
        for term in tfidf_dict:
            tfidf_dict[term] = tfidf_dict[term] / norm
        
    return tfidf_dict

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
    
    # Preprocess documents
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    
    # Generate and display n-grams for each document
    for n in range(1, 4):  # unigrams, bigrams, trigrams
        print(f"\n{n}-grams for all documents:")
        for doc_idx, doc in enumerate(preprocessed_docs):
            print(f"\nDocument {doc_idx+1} {n}-grams:")
            ngrams = generate_ngrams(doc, n)
            for i, ngram in enumerate(ngrams):  # Show all n-grams
                print(f"  {i+1}. {ngram}")
    
    # For TF-IDF calculation, use unigrams but ensure proper handling
    all_documents_ngrams = []
    for doc in preprocessed_docs:
        unigrams = generate_ngrams(doc, 1)
        all_documents_ngrams.append(unigrams)
    
    # Calculate IDF for all documents
    idf_dict = calculate_idf(all_documents_ngrams)
    
    # Calculate and display TF-IDF for each document
    print("\nTF-IDF Scores for all documents:")
    for i, doc_ngrams in enumerate(all_documents_ngrams):
        print(f"\nDocument {i+1}:")
        # Calculate term frequencies and TF-IDF scores
        doc_tf = calculate_tf(doc_ngrams)
        doc_tfidf = calculate_tfidf(doc_tf, idf_dict)
        
        # Create a list with unique terms to avoid duplicates
        seen_terms = set()
        unique_sorted_terms = []
        
        for term, score in sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True):
            if term not in seen_terms:
                seen_terms.add(term)
                unique_sorted_terms.append((term, score))
        
        # Display TF-IDF scores
        for term, score in unique_sorted_terms:
            print(f"  {term}: {score:.4f}")
        