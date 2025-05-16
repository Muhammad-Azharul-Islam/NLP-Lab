import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    # Initialize processing tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize into words
    words = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Tokenize into paragraphs (split by double newline)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    # Calculate statistics
    vocab = set(filtered_words)
    
    result = {
        'original_text': text,
        'sentences': sentences,
        'words': filtered_words,
        'stemmed_words': stemmed_words,
        'lemmatized_words': lemmatized_words,
        'paragraphs': paragraphs,
        'vocabulary_size': len(vocab),
        'word_count': len(filtered_words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs)
    }
    
    return result

if __name__ == "__main__":
    # Example text for preprocessing
    sample_text = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence. 
    It focuses on the interaction between computers and humans through natural language.
    
    The ultimate objective of NLP is to read, decipher, understand, and make sense of human language.
    Most NLP techniques rely on machine learning to derive meaning from human languages.
    
    Visit https://www.example.com for more information. Some special characters: !@#$%^&*()
    """
    
    result = preprocess_text(sample_text)
    
    # Print statistics
    print("Text Statistics:")
    print(f"Vocabulary Size: {result['vocabulary_size']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Sentence Count: {result['sentence_count']}")
    print(f"Paragraph Count: {result['paragraph_count']}")
    
    # Print processed text samples
    print("\nSample of Processed Text:")
    print("\nSentences (first 3):")
    for sentence in result['sentences'][:3]:
        print(f"- {sentence}")
    
    print("\nWords (first 10):")
    print(result['words'][:10])
    
    print("\nStemmed Words (first 10):")
    print(result['stemmed_words'][:10])
    
    print("\nLemmatized Words (first 10):")
    print(result['lemmatized_words'][:10]) 