#!/usr/bin/env python3
"""
Chess AI Training Script using Decision Transformer
Trains a model to predict next chess moves based on historical game data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import chess
import chess.pgn
from typing import List, Tuple
import json
import os
from datetime import datetime

class ChessGameDataset(Dataset):
    def __init__(self, games_data, max_sequence_length=50):
        self.games_data = games_data
        self.max_sequence_length = max_sequence_length
        self.sequences = self._prepare_sequences()
    
    def _move_to_token(self, move_str):
        """Convert move string to numerical token"""
        if not move_str or move_str == '':
            return 0
        
        # Simple tokenization: map move strings to integers
        move_hash = hash(move_str) % 4096  # Limit to reasonable vocab size
        return abs(move_hash) + 1  # Avoid 0 (reserved for padding)
    
    def _prepare_sequences(self):
        sequences = []
        
        for _, game in self.games_data.iterrows():
            moves_str = str(game['moves'])
            if pd.isna(moves_str) or moves_str == 'nan':
                continue
                
            moves = moves_str.split(' ')
            if len(moves) < 2:
                continue
            
            # Create sequences of moves for training
            move_tokens = [self._move_to_token(move) for move in moves]
            
            # Create sliding window sequences
            for i in range(1, min(len(move_tokens), self.max_sequence_length)):
                sequence = move_tokens[:i+1]
                if len(sequence) >= 2:  # Need at least 2 moves
                    # Pad sequence to max length
                    padded_sequence = sequence + [0] * (self.max_sequence_length - len(sequence))
                    sequences.append({
                        'sequence': padded_sequence[:self.max_sequence_length],
                        'length': min(len(sequence), self.max_sequence_length),
                        'target': move_tokens[i] if i < len(move_tokens) else 0
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return {
            'sequence': torch.tensor(item['sequence'], dtype=torch.long),
            'length': item['length'],
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

class DecisionTransformer(nn.Module):
    def __init__(self, vocab_size=4097, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=50):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, sequence, lengths=None):
        batch_size, seq_len = sequence.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(sequence)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = self.dropout(token_embeds + pos_embeds)
        
        # Create attention mask (optional - for padding)
        if lengths is not None:
            mask = torch.zeros(batch_size, seq_len, device=sequence.device, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, length:] = True
        else:
            mask = None
        
        # Transform
        output = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits

def load_and_preprocess_data(csv_path):
    """Load and preprocess chess games data"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} games")
    
    # Filter out games with no moves or invalid data
    df = df.dropna(subset=['moves'])
    df = df[df['moves'] != '']
    
    print(f"After filtering: {len(df)} games with valid moves")
    
    return df

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """Train the Decision Transformer model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            lengths = batch['length']
            
            # Forward pass
            logits = model(sequence, lengths)
            
            # Use the last non-padding position for prediction
            batch_size = sequence.shape[0]
            last_positions = [min(length - 1, logits.shape[1] - 1) for length in lengths]
            
            predictions = []
            targets = []
            for i, pos in enumerate(last_positions):
                if pos >= 0:
                    predictions.append(logits[i, pos])
                    targets.append(target[i])
            
            if predictions:
                pred_tensor = torch.stack(predictions)
                target_tensor = torch.stack(targets)
                
                loss = criterion(pred_tensor, target_tensor)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequence = batch['sequence'].to(device)
                target = batch['target'].to(device)
                lengths = batch['length']
                
                logits = model(sequence, lengths)
                
                batch_size = sequence.shape[0]
                last_positions = [min(length - 1, logits.shape[1] - 1) for length in lengths]
                
                predictions = []
                targets = []
                for i, pos in enumerate(last_positions):
                    if pos >= 0:
                        predictions.append(logits[i, pos])
                        targets.append(target[i])
                
                if predictions:
                    pred_tensor = torch.stack(predictions)
                    target_tensor = torch.stack(targets)
                    
                    loss = criterion(pred_tensor, target_tensor)
                    val_loss += loss.item()
                    val_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, 'best_chess_model.pth')
    
    return model

def predict_next_move(model, move_sequence, device='cpu'):
    """Predict next chess move given a sequence of moves"""
    model.eval()
    
    dataset = ChessGameDataset(pd.DataFrame([{'moves': ' '.join(move_sequence)}]))
    if len(dataset) == 0:
        return None
    
    with torch.no_grad():
        sample = dataset[0]
        sequence = sample['sequence'].unsqueeze(0).to(device)
        length = sample['length']
        
        logits = model(sequence, [length])
        
        # Get prediction from last position
        last_pos = min(length - 1, logits.shape[1] - 1)
        predictions = logits[0, last_pos]
        
        # Get top prediction
        predicted_token = torch.argmax(predictions).item()
        
        return predicted_token

def main():
    print("Chess AI Training with Decision Transformer")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    csv_path = 'games.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    df = load_and_preprocess_data(csv_path)
    
    # Create train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_df)} games")
    print(f"Validation set: {len(val_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Create datasets
    max_seq_len = 50
    train_dataset = ChessGameDataset(train_df, max_seq_len)
    val_dataset = ChessGameDataset(val_df, max_seq_len)
    test_dataset = ChessGameDataset(test_df, max_seq_len)
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DecisionTransformer(
        vocab_size=4097,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=max_seq_len
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    # Save final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'vocab_size': 4097,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'max_seq_len': max_seq_len
        },
        'training_info': {
            'total_games': len(df),
            'train_sequences': len(train_dataset),
            'timestamp': datetime.now().isoformat()
        }
    }, 'chess_decision_transformer.pth')
    
    print("\nTraining completed!")
    print("Model saved as 'chess_decision_transformer.pth'")
    print("Best model saved as 'best_chess_model.pth'")
    
    # Test prediction example
    print("\nTesting prediction...")
    example_moves = ['e4', 'e5', 'Nf3', 'd6']
    prediction = predict_next_move(trained_model, example_moves, device)
    print(f"Example moves: {' '.join(example_moves)}")
    print(f"Predicted next move token: {prediction}")

if __name__ == "__main__":
    main()