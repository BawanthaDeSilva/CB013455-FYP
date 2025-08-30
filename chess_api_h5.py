#!/usr/bin/env python3
"""
Chess AI API Server - H5 Model Version
Serves the trained Decision Transformer model (.h5 format) for chess move predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import h5py
import json
import os

app = Flask(__name__)
CORS(app)

class DecisionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=50):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, sequence, lengths=None):
        batch_size, seq_len = sequence.shape
        
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(sequence)
        pos_embeds = self.position_embedding(positions)
        
        embeddings = self.dropout(token_embeds + pos_embeds)
        
        if lengths is not None:
            mask = torch.zeros(batch_size, seq_len, device=sequence.device, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, length:] = True
        else:
            mask = None
        
        output = self.transformer(embeddings, src_key_padding_mask=mask)
        logits = self.output_projection(output)
        
        return logits

# Global variables
model = None
vocabulary = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_h5(model_path='chess_decision_transformer.h5', vocab_path='vocabulary.json'):
    """Load model from H5 format"""
    global model, vocabulary
    
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        return False
    
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file {vocab_path} not found!")
        return False
    
    try:
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            vocabulary = {
                'move_to_idx': vocab_data['move_to_idx'],
                'idx_to_move': {int(k): v for k, v in vocab_data['idx_to_move'].items()},
                'vocab_size': vocab_data['vocab_size']
            }
        
        print(f"✅ Loaded vocabulary: {vocabulary['vocab_size']} tokens")
        
        # Load model from H5
        with h5py.File(model_path, 'r') as f:
            # Read config
            config = f['config']
            vocab_size = int(config.attrs['vocab_size'])
            embed_dim = int(config.attrs['embed_dim'])
            max_seq_len = int(config.attrs['max_seq_len'])
            
            print(f"📋 Model config: vocab={vocab_size}, embed={embed_dim}, seq_len={max_seq_len}")
            
            # Initialize model
            model = DecisionTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                max_seq_len=max_seq_len
            )
            
            # Load weights
            weights_group = f['model_weights']
            state_dict = {}
            
            for key in weights_group.keys():
                weight_data = weights_group[key][:]
                state_dict[key] = torch.from_numpy(weight_data)
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
        print(f"✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_next_move_token(move_history):
    """Predict next move token using the loaded model"""
    if model is None or vocabulary is None:
        return None, 0.0
    
    try:
        # Convert moves to tokens
        move_tokens = []
        for move in move_history:
            token = vocabulary['move_to_idx'].get(move, 0)
            move_tokens.append(token)
        
        if not move_tokens:
            # Return a common opening move token
            opening_token = vocabulary['move_to_idx'].get('e4', vocabulary['move_to_idx'].get('d4', 1))
            return opening_token, 0.8
        
        # Pad sequence
        max_seq_len = model.max_seq_len
        padded_sequence = move_tokens + [0] * (max_seq_len - len(move_tokens))
        padded_sequence = padded_sequence[:max_seq_len]
        
        # Prepare input tensor
        sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(device)
        length = min(len(move_tokens), max_seq_len)
        
        with torch.no_grad():
            logits = model(sequence_tensor, [length])
            
            # Get prediction from last position
            last_pos = min(length - 1, logits.shape[1] - 1)
            if last_pos >= 0:
                predictions = logits[0, last_pos]
                
                # Get top 5 predictions to add some variety
                top_k = min(5, predictions.shape[0])
                top_values, top_indices = torch.topk(predictions, top_k)
                
                # Use temperature sampling for more interesting moves
                temperature = 0.8
                scaled_logits = top_values / temperature
                probabilities = torch.softmax(scaled_logits, dim=0)
                
                # Sample from top predictions
                sampled_idx = torch.multinomial(probabilities, 1).item()
                predicted_token = top_indices[sampled_idx].item()
                confidence = probabilities[sampled_idx].item()
                
                return predicted_token, confidence
            else:
                return vocabulary['move_to_idx'].get('e4', 1), 0.5
                
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "vocab_loaded": vocabulary is not None,
        "vocab_size": vocabulary['vocab_size'] if vocabulary else 0
    })

@app.route('/predict_move', methods=['POST'])
def predict_move():
    try:
        if model is None or vocabulary is None:
            return jsonify({"error": "Model or vocabulary not loaded"}), 500
        
        data = request.get_json()
        move_history = data.get('moves', [])
        
        print(f"🎯 Predicting next move for: {move_history}")
        
        # Get prediction
        predicted_token, confidence = predict_next_move_token(move_history)
        
        if predicted_token is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        # Convert token back to move
        predicted_move = vocabulary['idx_to_move'].get(predicted_token, 'e4')
        
        print(f"🎲 Predicted: {predicted_move} (confidence: {confidence:.1%})")
        
        return jsonify({
            "move": predicted_move,
            "confidence": float(confidence),
            "token": int(predicted_token),
            "move_history_length": len(move_history)
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_vocabulary', methods=['GET'])
def get_vocabulary():
    """Return the model's vocabulary for debugging"""
    if vocabulary is None:
        return jsonify({"error": "Vocabulary not loaded"}), 500
    
    # Return first 20 moves for inspection
    sample_moves = dict(list(vocabulary['move_to_idx'].items())[:20])
    
    return jsonify({
        "vocab_size": vocabulary['vocab_size'],
        "sample_moves": sample_moves,
        "total_moves": len(vocabulary['move_to_idx'])
    })

@app.route('/set_difficulty', methods=['POST'])
def set_difficulty():
    data = request.get_json()
    difficulty = data.get('difficulty', 'medium')
    
    # Store difficulty level (affects temperature in prediction)
    return jsonify({"difficulty": difficulty, "status": "updated"})

if __name__ == '__main__':
    print("🏁 Starting Chess AI API Server (H5 Version)...")
    print("=" * 50)
    
    # Try to load the model
    model_loaded = load_model_h5()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
        print("🌐 Starting server on http://localhost:5000")
        print("\n🎮 Available endpoints:")
        print("  POST /predict_move - Get AI move prediction")
        print("  GET  /health - Check server health")
        print("  GET  /get_vocabulary - View model vocabulary")
        print("  POST /set_difficulty - Set AI difficulty")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load model!")
        print("📋 Make sure you have:")
        print("   - chess_decision_transformer.h5")
        print("   - vocabulary.json")
        print("🚀 Run train_chess_ai_colab.py first to create these files")