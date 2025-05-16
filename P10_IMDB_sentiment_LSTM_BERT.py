import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re
import nltk
import time
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Download NLTK data
nltk.download('punkt', quiet=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # BERT tokenization
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
        
        # Return text and label for Word2Vec + LSTM
        return text, label

def preprocess_text(text):
    """
    Preprocess text for model input
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens
    """
    return word_tokenize(text)

def load_imdb_data(sample=True, sample_size=1000):
    """
    Load IMDB dataset (this is a simulation since we don't have actual data)
    
    Args:
        sample (bool): Whether to return a sample of the data
        sample_size (int): Size of the sample
        
    Returns:
        tuple: (texts, labels)
    """
    print("Note: This is a simulation. In a real scenario, you would load the actual IMDB dataset.")
    
    # Simulated positive reviews
    positive_reviews = [
        "This movie was fantastic! I really enjoyed it and would recommend it to anyone.",
        "One of the best films I've seen in years. The acting was superb.",
        "Great plot, amazing characters, and beautiful cinematography. A masterpiece!",
        "I was pleasantly surprised by how good this film was. Will definitely watch again.",
        "The director did an amazing job. The story was compelling from start to finish."
    ]
    
    # Simulated negative reviews
    negative_reviews = [
        "This was a terrible movie. I was bored throughout the entire thing.",
        "Waste of time and money. The plot made no sense at all.",
        "Awful acting and a predictable story. Would not recommend.",
        "I've seen better films made by students. Very disappointing.",
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

class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, embedding_weights=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Initialize with pre-trained embeddings if provided
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text: [batch size, seq len]
        
        # Apply embedding layer
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                          batch_first=True, enforce_sorted=False)
        
        # Apply LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the final hidden state for classification
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # Apply fully connected layer
        return self.fc(hidden)

def create_word2vec_embedding_matrix(word2vec_model, word_to_idx, embedding_dim):
    """
    Create an embedding matrix from Word2Vec model
    
    Args:
        word2vec_model: Trained Word2Vec model
        word_to_idx (dict): Mapping from words to indices
        embedding_dim (int): Dimensionality of embeddings
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in word_to_idx.items():
        try:
            embedding_matrix[idx] = word2vec_model.wv[word]
        except KeyError:
            # Random initialization for unknown words
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    return embedding_matrix

def train_word2vec_lstm(texts, labels, device=torch.device('cpu')):
    """
    Train a Word2Vec + LSTM model for sentiment analysis
    
    Args:
        texts (list): List of preprocessed text samples
        labels (list): List of labels
        device: Torch device to use
        
    Returns:
        model: Trained model
    """
    print("\nTraining Word2Vec + LSTM model...")
    
    # Tokenize texts
    tokenized_texts = [tokenize_text(text) for text in texts]
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    print("Word2Vec model trained successfully")
    
    # Create vocabulary from Word2Vec model
    word_to_idx = {word: idx+2 for idx, word in enumerate(word2vec_model.wv.index_to_key)}
    word_to_idx['<pad>'] = 0
    word_to_idx['<unk>'] = 1
    
    # Convert texts to sequences of indices
    sequences = []
    for tokens in tokenized_texts:
        seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]
        sequences.append(seq)
    
    # Calculate sequence lengths
    seq_lengths = torch.LongTensor([len(seq) for seq in sequences])
    
    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = torch.LongTensor(seq)
    
    # Create embedding matrix from Word2Vec
    embedding_matrix = create_word2vec_embedding_matrix(word2vec_model, word_to_idx, 100)
    
    # Split data
    X_train, X_val, y_train, y_val, train_lens, val_lens = train_test_split(
        padded_sequences, torch.tensor(labels), seq_lengths, test_size=0.2, random_state=42
    )
    
    # Define model parameters
    vocab_size = len(word_to_idx)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2  # Binary classification
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = word_to_idx['<pad>']
    
    # Initialize model
    model = LSTMSentimentModel(
        vocab_size, embedding_dim, hidden_dim, output_dim, 
        n_layers, bidirectional, dropout, pad_idx, embedding_matrix
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training parameters
    batch_size = 64
    epochs = 5
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        # Process in batches
        for i in range(0, len(X_train), batch_size):
            # Get batch
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            batch_lens = train_lens[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X, batch_lens)
            
            # Calculate loss
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(predictions, dim=1)
            correct = (predictions == batch_y).float().sum()
            acc = correct / len(batch_y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        # Calculate epoch metrics
        epoch_loss /= (len(X_train) // batch_size + 1)
        epoch_acc /= (len(X_train) // batch_size + 1)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for i in range(0, len(X_val), batch_size):
            # Get batch
            batch_X = X_val[i:i+batch_size].to(device)
            batch_y = y_val[i:i+batch_size].numpy()
            batch_lens = val_lens[i:i+batch_size]
            
            # Forward pass
            predictions = model(batch_X, batch_lens)
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(batch_y)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(all_labels, all_preds))
    
    return model

def train_bert(texts, labels, device=torch.device('cpu')):
    """
    Train a BERT model for sentiment analysis
    
    Args:
        texts (list): List of preprocessed text samples
        labels (list): List of labels
        device: Torch device to use
        
    Returns:
        model: Trained model
    """
    print("\nTraining BERT model...")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
    val_dataset = IMDbDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16  # Smaller batch size for BERT due to memory constraints
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    total_steps = len(train_loader) * 3  # epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training parameters
    epochs = 3
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # Calculate epoch loss
        epoch_loss /= len(train_loader)
        print(f"Training loss: {epoch_loss:.4f}")
        
        # Evaluate on validation set
        model.eval()
        val_accuracy = 0
        val_loss = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
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
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
            
            # Calculate metrics
            val_loss /= len(val_loader)
            accuracy = accuracy_score(all_labels, all_preds)
            
            print(f"Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            print(classification_report(all_labels, all_preds))
    
    return model, tokenizer

if __name__ == "__main__":
    print("IMDB Sentiment Analysis with Word2Vec + LSTM and BERT")
    
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
        print(f"Label: {labels[i]}")
    
    # Part A: Word2Vec + LSTM
    start_time = time.time()
    lstm_model = train_word2vec_lstm(texts, labels, device)
    lstm_time = time.time() - start_time
    print(f"Word2Vec + LSTM training time: {lstm_time:.2f} seconds")
    
    # Part B: BERT
    start_time = time.time()
    bert_model, bert_tokenizer = train_bert(texts, labels, device)
    bert_time = time.time() - start_time
    print(f"BERT training time: {bert_time:.2f} seconds")
    
    # Compare training times
    plt.figure(figsize=(8, 6))
    plt.bar(['Word2Vec + LSTM', 'BERT'], [lstm_time, bert_time])
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.savefig('training_time_comparison.png')
    plt.close()
    
    print("Training time comparison chart saved as 'training_time_comparison.png'")
    
    # Note on production implementation
    print("\nNote: For a real implementation, you would:")
    print("1. Use the actual IMDB dataset (or a large sentiment dataset)")
    print("2. Train for more epochs with early stopping")
    print("3. Implement more thorough evaluation methods")
    print("4. Possibly use pre-trained Word2Vec embeddings like GloVe") 