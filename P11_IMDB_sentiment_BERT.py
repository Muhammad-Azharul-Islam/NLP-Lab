import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    """
    Preprocess text by removing HTML tags, special characters, 
    and extra whitespace
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_imdb_data(sample=True, sample_size=1000):
    """
    Load simulated IMDB dataset
    
    Args:
        sample (bool): Whether to create a smaller sample
        sample_size (int): Size of sample to create
        
    Returns:
        tuple: (texts, labels)
    """
    print("Note: This is a simulation. In a real scenario, download and load the actual IMDB dataset.")
    
    # Simulated positive reviews
    positive_reviews = [
        "This movie was absolutely fantastic! I really enjoyed it and would recommend it to anyone.",
        "One of the best films I've seen in years. The acting was superb and the plot was engaging.",
        "Great plot, amazing characters, and beautiful cinematography. A true masterpiece!",
        "I was pleasantly surprised by how good this film was. Will definitely watch it again.",
        "The director did an amazing job. The story was compelling from start to finish."
    ]
    
    # Simulated negative reviews
    negative_reviews = [
        "This was a terrible movie. I was bored throughout the entire thing and regret watching it.",
        "Waste of time and money. The plot made no sense at all and the acting was terrible.",
        "Awful acting and a predictable story. Would not recommend to anyone.",
        "I've seen better films made by students. Very disappointing and a waste of potential.",
        "The worst movie I've seen this year. Save your money and skip this one."
    ]
    
    # Simulate a larger dataset by duplicating and slightly modifying reviews
    all_texts = []
    all_labels = []
    
    for i in range(sample_size // 10):
        for review in positive_reviews:
            all_texts.append(review + f" {i}")
            all_labels.append(1)
        
        for review in negative_reviews:
            all_texts.append(review + f" {i}")
            all_labels.append(0)
    
    # Preprocess all texts
    all_texts = [preprocess_text(text) for text in all_texts]
    
    # Shuffle the data
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    return all_texts, all_labels

def train_bert_model(train_dataloader, eval_dataloader, model, device, epochs=4):
    """
    Train the BERT model
    
    Args:
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        model: Pre-trained BERT model
        device: Device to train on (CPU or GPU)
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (trained model, training stats)
    """
    # Move model to the specified device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training stats
    training_stats = []
    
    # Training loop
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')
        print('Training...')
        
        # Reset metrics
        total_train_loss = 0
        model.train()
        
        # Training loop with progress bar
        progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
        for batch in progress_bar:
            # Clear gradients
            model.zero_grad()
            
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradient norms to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # Calculate average loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")
        
        # Evaluation
        print("Running Validation...")
        model.eval()
        
        total_eval_loss = 0
        all_preds = []
        all_labels = []
        
        # Evaluate without computing gradients
        for batch in tqdm(eval_dataloader, desc="Validation", leave=True):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass, no gradient
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_eval_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            label_ids = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(label_ids)
        
        # Calculate metrics
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Validation Loss: {avg_eval_loss:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
        
        # Store stats for this epoch
        training_stats.append({
            'epoch': epoch + 1,
            'training_loss': avg_train_loss,
            'valid_loss': avg_eval_loss,
            'accuracy': accuracy
        })
    
    return model, training_stats

def plot_training_stats(training_stats):
    """
    Plot training and validation metrics
    
    Args:
        training_stats (list): List of dictionaries containing training statistics
    """
    # Convert to DataFrame
    df_stats = pd.DataFrame(training_stats)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_stats['epoch'], df_stats['training_loss'], 'b-o', label='Training Loss')
    plt.plot(df_stats['epoch'], df_stats['valid_loss'], 'g-o', label='Validation Loss')
    plt.plot(df_stats['epoch'], df_stats['accuracy'], 'r-o', label='Accuracy')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('bert_training_metrics.png')
    plt.close()
    
    print("Training metrics plot saved as 'bert_training_metrics.png'")

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.savefig('bert_confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix plot saved as 'bert_confusion_matrix.png'")

def evaluate_model(model, test_dataloader, device):
    """
    Evaluate trained model on test data
    
    Args:
        model: Trained BERT model
        test_dataloader: Test data loader
        device: Device to evaluate on
        
    Returns:
        tuple: (accuracy, predictions, true labels)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tracking variables
    all_preds = []
    all_labels = []
    
    # Evaluate without computing gradients
    for batch in tqdm(test_dataloader, desc="Testing"):
        # Get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass, no gradient
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predictions
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        label_ids = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(label_ids)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy, all_preds, all_labels

def analyze_sentiment(texts, model, tokenizer, device):
    """
    Analyze sentiment of new texts using the trained model
    
    Args:
        texts (list): List of texts to analyze
        model: Trained BERT model
        tokenizer: BERT tokenizer
        device: Device to run inference on
        
    Returns:
        list: List of sentiment predictions (0 for negative, 1 for positive)
    """
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a dataset and dataloader for the new texts
    # We'll use dummy labels that won't be used
    dummy_labels = [0] * len(preprocessed_texts)
    dataset = IMDbDataset(preprocessed_texts, dummy_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # Get predictions
    predictions = []
    confidence_scores = []
    
    for batch in dataloader:
        # Get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass, no gradient
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predictions and confidence scores
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        batch_confidence = probs.max(dim=1)[0].cpu().numpy()
        
        predictions.extend(batch_preds)
        confidence_scores.extend(batch_confidence)
    
    result = []
    for i in range(len(texts)):
        sentiment = "Positive" if predictions[i] == 1 else "Negative"
        result.append({
            'text': texts[i],
            'sentiment': sentiment,
            'confidence': confidence_scores[i]
        })
    
    return result

if __name__ == "__main__":
    print("IMDB Sentiment Analysis using BERT")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    texts, labels = load_imdb_data(sample=True, sample_size=1000)
    print(f"Loaded {len(texts)} samples")
    
    # Display sample data
    print("\nSample data:")
    for i in range(3):
        print(f"Text: {texts[i][:100]}...")
        print(f"Label: {labels[i]} ({'Positive' if labels[i] == 1 else 'Negative'})")
    
    # Split data into training, validation, and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Load pre-trained BERT model and tokenizer
    print("\nLoading pre-trained BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Create datasets
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
    val_dataset = IMDbDataset(val_texts, val_labels, tokenizer)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    # Train model
    print("\nTraining BERT model...")
    start_time = time.time()
    
    model, training_stats = train_bert_model(
        train_dataloader,
        val_dataloader,
        model,
        device,
        epochs=3
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training metrics
    plot_training_stats(training_stats)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_preds, test_labels = evaluate_model(model, test_dataloader, device)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds)
    
    # Analyze new texts
    print("\nAnalyzing sentiment of new reviews...")
    new_reviews = [
        "This movie was amazing! The plot was engaging and the characters were well-developed.",
        "Terrible film. I fell asleep halfway through and didn't bother to finish it.",
        "Mixed feelings about this one. Some parts were good, but others dragged on too long.",
        "I would recommend this film to anyone who enjoys thoughtful dramas.",
        "The acting was good but the script needed work. Overall an average experience."
    ]
    
    results = analyze_sentiment(new_reviews, model, tokenizer, device)
    
    print("\nSentiment Analysis Results:")
    for result in results:
        print(f"Text: {result['text'][:70]}...")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
        print()
    
    # Save model and tokenizer
    model_save_path = 'bert_imdb_sentiment'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model and tokenizer saved to '{model_save_path}'")
    
    print("\nNote: For a real implementation, you would:")
    print("1. Use the actual IMDB dataset")
    print("2. Possibly try different BERT variants (BERT-large, DistilBERT, etc.)")
    print("3. Experiment with more hyperparameter tuning")
    print("4. Implement cross-validation for more robust evaluation") 