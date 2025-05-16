import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data():
    """
    Load 20 Newsgroups dataset
    
    Returns:
        tuple: (training data, test data, training target, test target)
    """
    # Get training and test data
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    # Load training data
    newsgroups_train = fetch_20newsgroups(subset='train', 
                                          categories=categories,
                                          shuffle=True, 
                                          random_state=42)
    
    # Load test data
    newsgroups_test = fetch_20newsgroups(subset='test', 
                                         categories=categories,
                                         shuffle=True, 
                                         random_state=42)
    
    return (newsgroups_train.data, newsgroups_test.data, 
            newsgroups_train.target, newsgroups_test.target,
            newsgroups_train.target_names)

def train_naive_bayes(X_train, y_train, ngram_range=(1, 1), alpha=1.0):
    """
    Train a Naive Bayes classifier on text data
    
    Args:
        X_train (list): Training text data
        y_train (list): Training labels
        ngram_range (tuple): Range of n-grams to use
        alpha (float): Additive (Laplace/Lidstone) smoothing parameter
        
    Returns:
        object: Trained classifier
    """
    # Create a pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=alpha)),
    ])
    
    # Train the classifier
    text_clf.fit(X_train, y_train)
    
    return text_clf

def evaluate_classifier(clf, X_test, y_test, target_names):
    """
    Evaluate the classifier on test data
    
    Args:
        clf: Trained classifier
        X_test (list): Test text data
        y_test (list): Test labels
        target_names (list): Names of target classes
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

def grid_search_nb(X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for Naive Bayes
    
    Args:
        X_train (list): Training text data
        y_train (list): Training labels
        
    Returns:
        object: Best model
    """
    # Create a pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    
    # Define parameters to search
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search

if __name__ == "__main__":
    # Load data
    print("Loading dataset...")
    X_train, X_test, y_train, y_test, target_names = load_data()
    
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Categories: {target_names}")
    
    # Train a basic Naive Bayes model with unigrams
    print("\nTraining Naive Bayes classifier with unigrams...")
    nb_classifier = train_naive_bayes(X_train, y_train, ngram_range=(1, 1))
    
    # Evaluate the classifier
    print("\nEvaluating the basic classifier...")
    evaluation = evaluate_classifier(nb_classifier, X_test, y_test, target_names)
    
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print("\nClassification Report:")
    print(evaluation['classification_report'])
    
    print("\nConfusion Matrix:")
    print(evaluation['confusion_matrix'])
    
    # Try different n-gram ranges
    print("\nComparing different n-gram configurations:")
    for ngram_range in [(1, 1), (1, 2), (2, 2)]:
        print(f"\nTraining with n-gram range {ngram_range}...")
        nb_classifier = train_naive_bayes(X_train, y_train, ngram_range=ngram_range)
        evaluation = evaluate_classifier(nb_classifier, X_test, y_test, target_names)
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
    
    # Perform grid search to find the best parameters
    print("\nPerforming grid search to find optimal parameters...")
    grid_search = grid_search_nb(X_train, y_train)
    
    # Print the best parameters and score
    print("\nBest parameters:")
    print(grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate the best model
    print("\nEvaluating the best model on test data...")
    best_model = grid_search.best_estimator_
    best_evaluation = evaluate_classifier(best_model, X_test, y_test, target_names)
    
    print(f"Best model accuracy: {best_evaluation['accuracy']:.4f}")
    print("\nClassification Report for Best Model:")
    print(best_evaluation['classification_report']) 