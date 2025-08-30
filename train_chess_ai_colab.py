#!/usr/bin/env python3
"""
Chess AI Training Script for Google Colab using Decision Transformer
Trains a model to predict next chess moves and saves as .h5 format
"""

# Install required packages in Colab
import subprocess
import sys

def install_packages():
    """Install required packages in Google Colab"""
    packages = [
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'python-chess',
        'tensorflow',
        'h5py'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")

# Run installation
print("Installing required packages...")
install_packages()

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
import h5py
import tensorflow as tf

# Upload instructions for Colab
print("""
📋 GOOGLE COLAB SETUP INSTRUCTIONS:
1. Upload your 'games.csv' file to this Colab session
2. Click the folder icon on the left sidebar
3. Drag and drop your games.csv file into the file browser
4. Run this cell to start training
""")

class ChessGameDataset(Dataset):
    def __init__(self, games_data, max_sequence_length=50):
        self.games_data = games_data
        self.max_sequence_length = max_sequence_length
        self.move_to_idx = {}
        self.idx_to_move = {}
        self.vocab_size = 1
        self._build_vocabulary()
        self.sequences = self._prepare_sequences()
    
    def _build_vocabulary(self):
        """Build vocabulary from all moves in dataset"""
        all_moves = set()
        
        for _, game in self.games_data.iterrows():
            moves_str = str(game['moves'])
            if pd.isna(moves_str) or moves_str == 'nan':
                continue
                
            moves = moves_str.split(' ')
            all_moves.update(moves)
        
        # Remove empty strings
        all_moves.discard('')
        all_moves = sorted(list(all_moves))
        
        # Build mappings (0 reserved for padding)
        self.move_to_idx = {move: idx + 1 for idx, move in enumerate(all_moves)}
        self.idx_to_move = {idx + 1: move for idx, move in enumerate(all_moves)}
        self.idx_to_move[0] = '<PAD>'
        self.vocab_size = len(all_moves) + 1
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def _move_to_token(self, move_str):
        """Convert move string to numerical token"""
        return self.move_to_idx.get(move_str, 0)
    
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
            move_tokens = [self._move_to_token(move) for move in moves if move]
            
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
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=50):
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
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print("Please upload games.csv to this Colab session")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} games")
    
    # Filter out games with no moves or invalid data
    df = df.dropna(subset=['moves'])
    df = df[df['moves'] != '']
    
    print(f"✓ After filtering: {len(df)} games with valid moves")
    
    return df

def save_model_h5(model, vocab_mappings, filepath='chess_model.h5'):
    """Save PyTorch model to HDF5 format"""
    print(f"Saving model to {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Save model config
        config_group = f.create_group('config')
        config_group.attrs['vocab_size'] = model.vocab_size
        config_group.attrs['embed_dim'] = model.embed_dim
        config_group.attrs['max_seq_len'] = model.max_seq_len
        config_group.attrs['model_type'] = 'DecisionTransformer'
        
        # Save vocabulary mappings
        vocab_group = f.create_group('vocabulary')
        
        # Convert mappings to arrays for HDF5 storage
        moves = list(vocab_mappings['idx_to_move'].values())
        indices = list(vocab_mappings['idx_to_move'].keys())
        
        vocab_group.create_dataset('moves', data=[m.encode('utf-8') for m in moves])
        vocab_group.create_dataset('indices', data=indices)
        
        # Save model weights
        weights_group = f.create_group('model_weights')
        state_dict = model.state_dict()
        
        for key, tensor in state_dict.items():
            weights_group.create_dataset(key, data=tensor.cpu().numpy())
    
    print(f"✓ Model saved to {filepath}")

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """Train the Decision Transformer model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    best_val_loss = float('inf')
    
    print(f"🚀 Starting training on {device}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
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
        
        print(f"🎯 Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"💾 New best model! Val Loss: {avg_val_loss:.4f}")
    
    return model

def main():
    print("🏁 Chess AI Training with Decision Transformer - Google Colab Version")
    print("=" * 70)
    
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        IN_COLAB = True
    except ImportError:
        print("⚠️  Not running in Google Colab")
        IN_COLAB = False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # Load data
    csv_path = 'games.csv'
    df = load_and_preprocess_data(csv_path)
    
    if df is None:
        return
    
    # Create train/test split
    print("\n📊 Creating train/validation/test splits...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    print(f"📈 Train set: {len(train_df)} games")
    print(f"📊 Validation set: {len(val_df)} games")  
    print(f"🧪 Test set: {len(test_df)} games")
    
    # Create datasets
    max_seq_len = 50
    print(f"\n🔄 Processing sequences (max length: {max_seq_len})...")
    
    train_dataset = ChessGameDataset(train_df, max_seq_len)
    val_dataset = ChessGameDataset(val_df, max_seq_len)
    test_dataset = ChessGameDataset(test_df, max_seq_len)
    
    print(f"🎯 Training sequences: {len(train_dataset):,}")
    print(f"🎯 Validation sequences: {len(val_dataset):,}")
    print(f"🎯 Test sequences: {len(test_dataset):,}")
    
    if len(train_dataset) == 0:
        print("❌ No valid training sequences found!")
        return
    
    # Get vocabulary size from dataset
    vocab_size = train_dataset.vocab_size
    print(f"📝 Vocabulary size: {vocab_size}")
    
    # Create data loaders
    batch_size = 32 if device.type == 'cuda' else 16
    print(f"📦 Batch size: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n🧠 Initializing Decision Transformer...")
    model = DecisionTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=max_seq_len
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Train model
    print(f"\n🚀 Starting training...")
    num_epochs = 20 if device.type == 'cuda' else 10
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
    
    # Prepare vocabulary mappings for saving
    vocab_mappings = {
        'move_to_idx': train_dataset.move_to_idx,
        'idx_to_move': train_dataset.idx_to_move,
        'vocab_size': vocab_size
    }
    
    # Save model in .h5 format
    print(f"\n💾 Saving model...")
    save_model_h5(trained_model, vocab_mappings, 'chess_decision_transformer.h5')
    
    # Also save vocabulary as separate JSON for easy loading
    with open('vocabulary.json', 'w') as f:
        # Convert integer keys to strings for JSON
        vocab_for_json = {
            'move_to_idx': vocab_mappings['move_to_idx'],
            'idx_to_move': {str(k): v for k, v in vocab_mappings['idx_to_move'].items()},
            'vocab_size': vocab_size
        }
        json.dump(vocab_for_json, f, indent=2)
    
    print("✅ Training completed!")
    print("📁 Files created:")
    print("   - chess_decision_transformer.h5 (main model)")
    print("   - vocabulary.json (move mappings)")
    
    if IN_COLAB:
        print(f"\n📥 DOWNLOAD INSTRUCTIONS:")
        print("1. Right-click on 'chess_decision_transformer.h5' in the file browser")
        print("2. Select 'Download' to save to your computer")
        print("3. Also download 'vocabulary.json'")
        print("4. Upload both files to your local chess project")
    
    # Test prediction example
    print(f"\n🧪 Testing prediction...")
    try:
        model.eval()
        example_moves = ['e4', 'e5', 'Nf3', 'd6']
        
        # Convert to tokens
        example_tokens = [train_dataset._move_to_token(move) for move in example_moves]
        padded = example_tokens + [0] * (max_seq_len - len(example_tokens))
        
        with torch.no_grad():
            sequence_tensor = torch.tensor([padded[:max_seq_len]], dtype=torch.long).to(device)
            logits = model(sequence_tensor, [len(example_tokens)])
            
            last_pos = min(len(example_tokens) - 1, logits.shape[1] - 1)
            predictions = logits[0, last_pos]
            top_prediction = torch.argmax(predictions).item()
            confidence = torch.softmax(predictions, dim=0)[top_prediction].item()
            
            predicted_move = train_dataset.idx_to_move.get(top_prediction, 'Unknown')
            
            print(f"🎲 Example sequence: {' '.join(example_moves)}")
            print(f"🎯 Predicted next move: {predicted_move}")
            print(f"📊 Confidence: {confidence:.1%}")
            
    except Exception as e:
        print(f"⚠️  Prediction test failed: {e}")

if __name__ == "__main__":
    main()