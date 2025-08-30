#!/usr/bin/env python3
"""
Chess AI API Server using OpenAI GPT for intelligent move predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ChessAI:
    def __init__(self):
        self.board = chess.Board()
        self.difficulty = 'medium'
        
    def reset_board(self):
        self.board = chess.Board()
        
    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        
    def get_board_fen(self):
        return self.board.fen()
        
    def make_move(self, move_str):
        """Make a move on the internal board"""
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except:
            return False
            
    def get_legal_moves(self):
        """Get all legal moves in UCI format"""
        return [move.uci() for move in self.board.legal_moves]
        
    def convert_square_id_to_uci(self, from_id, to_id):
        """Convert square IDs (0-63) to UCI notation"""
        def id_to_square(square_id):
            file = chr(ord('a') + (square_id % 8))
            rank = str((square_id // 8) + 1)
            return file + rank
            
        from_square = id_to_square(int(from_id))
        to_square = id_to_square(int(to_id))
        return from_square + to_square

# Global chess AI instance
chess_ai = ChessAI()

async def get_openai_chess_move(board_fen, move_history, difficulty='medium'):
    """Get chess move prediction from OpenAI with thorough board analysis"""
    
    # Analyze current board position
    board = chess.Board(board_fen)
    
    # Get game phase and material analysis
    material_count = len([p for p in board.piece_map().values()])
    game_phase = "opening" if len(move_history) < 10 else "middlegame" if material_count > 10 else "endgame"
    
    # Check for tactical opportunities
    in_check = board.is_check()
    legal_moves_count = len(list(board.legal_moves))
    
    difficulty_prompts = {
        'easy': "You're a beginner chess player. Focus on basic tactics and piece safety.",
        'medium': "You're an intermediate chess player. Consider tactics, positional play, and strategic goals.",
        'hard': "You're an expert chess player. Perform deep analysis including tactics, strategy, pawn structure, king safety, and long-term planning."
    }
    
    analysis_context = f"""
    BOARD ANALYSIS:
    - Game Phase: {game_phase}
    - Material Count: {material_count} pieces remaining
    - King in Check: {in_check}
    - Legal Moves Available: {legal_moves_count}
    - Turn: {'White' if board.turn else 'Black'}
    
    TACTICAL ANALYSIS:
    - Look for checks, captures, and threats
    - Evaluate piece safety and hanging pieces
    - Consider pawn structure and weaknesses
    - Assess king safety and potential attacks
    """
    
    system_prompt = f"""You are a chess AI. {difficulty_prompts[difficulty]}
    
    ANALYZE THE POSITION THOROUGHLY:
    {analysis_context}
    
    Rules:
    1. ONLY respond with a valid UCI move (e.g., 'e2e4', 'g1f3', 'e1g1')
    2. The move must be legal for the current board position
    3. Do not include any explanation or extra text
    4. If castling, use UCI format: e1g1 (king side), e1c1 (queen side)
    5. Consider the current board position and all pieces carefully
    
    Current board FEN: {board_fen}
    Move history: {' '.join(move_history[-10:]) if move_history else 'Game start'}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"What's your next move? Current position: {board_fen}"}
            ],
            max_tokens=10,
            temperature=0.7 if difficulty == 'easy' else 0.3
        )
        
        move = response.choices[0].message.content.strip()
        
        # Validate move format (basic check)
        if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', move):
            return move
        else:
            # Fallback to random legal move if OpenAI response is invalid
            legal_moves = chess_ai.get_legal_moves()
            return legal_moves[0] if legal_moves else "e2e4"
            
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # Fallback to random legal move
        legal_moves = chess_ai.get_legal_moves()
        return legal_moves[0] if legal_moves else "e2e4"

@app.route('/health', methods=['GET'])
def health_check():
    api_key_configured = bool(os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here')
    return jsonify({
        "status": "healthy", 
        "openai_configured": api_key_configured,
        "current_fen": chess_ai.get_board_fen()
    })

@app.route('/predict_move', methods=['POST'])
def predict_move():
    try:
        data = request.get_json()
        move_history = data.get('moves', [])
        board_state = data.get('board_state', None)
        
        # Update internal board with move history
        if 'reset' in data and data['reset']:
            chess_ai.reset_board()
            
        # Sync moves with internal board
        current_fen = chess_ai.get_board_fen()
        
        # Get AI move prediction using OpenAI
        import asyncio
        predicted_move = asyncio.run(get_openai_chess_move(
            current_fen, 
            move_history, 
            chess_ai.difficulty
        ))
        
        # Validate the move is legal
        legal_moves = chess_ai.get_legal_moves()
        if predicted_move not in legal_moves and legal_moves:
            predicted_move = legal_moves[0]  # Fallback to first legal move
            
        # Calculate confidence based on difficulty
        confidence_map = {'easy': 0.6, 'medium': 0.8, 'hard': 0.95}
        confidence = confidence_map.get(chess_ai.difficulty, 0.8)
        
        return jsonify({
            "move": predicted_move,
            "confidence": float(confidence),
            "legal_moves_count": len(legal_moves),
            "current_fen": current_fen
        })
        
    except Exception as e:
        print(f"Error in predict_move: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/make_move', methods=['POST'])
def make_move():
    """Record a move made by the human player"""
    try:
        data = request.get_json()
        move_uci = data.get('move')
        
        print(f"Received move: {move_uci}")
        print(f"Current board: {chess_ai.get_board_fen()}")
        print(f"Legal moves: {chess_ai.get_legal_moves()[:5]}...")  # Show first 5
        
        if chess_ai.make_move(move_uci):
            print(f"Move executed successfully: {move_uci}")
            return jsonify({
                "success": True,
                "current_fen": chess_ai.get_board_fen()
            })
        else:
            print(f"Invalid move rejected: {move_uci}")
            return jsonify({
                "error": f"Invalid move: {move_uci}", 
                "legal_moves": chess_ai.get_legal_moves()[:10]
            }), 400
            
    except Exception as e:
        print(f"Error in make_move: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_game', methods=['POST'])
def reset_game():
    """Reset the chess game"""
    chess_ai.reset_board()
    return jsonify({
        "success": True,
        "current_fen": chess_ai.get_board_fen()
    })

@app.route('/set_difficulty', methods=['POST'])
def set_difficulty():
    data = request.get_json()
    difficulty = data.get('difficulty', 'medium')
    chess_ai.set_difficulty(difficulty)
    
    return jsonify({"difficulty": difficulty, "status": "updated"})

@app.route('/analyze_position', methods=['POST'])
def analyze_position():
    """Detailed board position analysis"""
    try:
        data = request.get_json()
        move_history = data.get('moves', [])
        
        # Sync internal board with move history
        chess_ai.reset_board()
        for move in move_history:
            chess_ai.make_move(move)
        
        board = chess_ai.board
        
        # Comprehensive position analysis
        analysis = {
            "game_phase": get_game_phase(board, len(move_history)),
            "material_balance": get_material_balance(board),
            "king_safety": assess_king_safety(board),
            "tactical_opportunities": find_tactical_opportunities(board),
            "positional_factors": assess_positional_factors(board),
            "best_move": "",
            "evaluation": 0.0
        }
        
        # Get best move with analysis
        import asyncio
        best_move = asyncio.run(get_openai_chess_move(
            board.fen(), 
            move_history, 
            chess_ai.difficulty
        ))
        
        analysis["best_move"] = best_move
        analysis["current_fen"] = board.fen()
        analysis["legal_moves_count"] = len(list(board.legal_moves))
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error in analyze_position: {e}")
        return jsonify({"error": str(e)}), 500

def get_game_phase(board, move_count):
    material_count = len([p for p in board.piece_map().values()])
    if move_count < 10:
        return "opening"
    elif material_count > 10:
        return "middlegame" 
    else:
        return "endgame"

def get_material_balance(board):
    values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    white_material = sum(values.get(str(piece).lower(), 0) for piece in board.piece_map().values() if piece.color)
    black_material = sum(values.get(str(piece).lower(), 0) for piece in board.piece_map().values() if not piece.color)
    return {"white": white_material, "black": black_material, "difference": white_material - black_material}

def assess_king_safety(board):
    return {
        "white_in_check": board.is_check() and board.turn,
        "black_in_check": board.is_check() and not board.turn,
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate()
    }

def find_tactical_opportunities(board):
    opportunities = []
    if board.is_check():
        opportunities.append("King in check - must respond")
    
    # Look for hanging pieces (simplified)
    for square, piece in board.piece_map().items():
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        if len(attackers) > len(defenders):
            opportunities.append(f"Hanging {piece.symbol()} on {chess.square_name(square)}")
    
    return opportunities[:3]  # Limit to top 3

def assess_positional_factors(board):
    return {
        "center_control": "Evaluate center square control",
        "pawn_structure": "Assess pawn weaknesses",
        "piece_activity": "Evaluate piece mobility"
    }

if __name__ == '__main__':
    print("Starting Chess AI API Server with OpenAI integration...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("⚠️  WARNING: OpenAI API key not configured!")
        print("Please set your OpenAI API key in the .env file")
        print("OPENAI_API_KEY=your_actual_api_key")
    else:
        print("✅ OpenAI API key configured")
    
    print("Starting server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)