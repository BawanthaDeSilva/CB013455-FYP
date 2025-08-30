# Chess AI Setup Instructions

## 🎯 Complete Setup Guide

### Step 1: Train Model in Google Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload training script**:
   - Upload `train_chess_ai_colab.py` to your Colab session
   - Or copy-paste the code into a new notebook

3. **Upload dataset**:
   - Upload `games.csv` to the Colab file browser
   - Click the folder icon on the left sidebar
   - Drag and drop `games.csv` into the files

4. **Run training**:
   ```python
   # In Colab cell:
   exec(open('train_chess_ai_colab.py').read())
   ```

5. **Download trained model**:
   - After training completes, download these files:
     - `chess_decision_transformer.h5` (main model)
     - `vocabulary.json` (move mappings)

### Step 2: Setup Local API Server

1. **Place model files**:
   - Put `chess_decision_transformer.h5` in your project folder
   - Put `vocabulary.json` in your project folder

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start API server**:
   ```bash
   python chess_api_h5.py
   ```

### Step 3: Play Chess vs AI

1. **Open the game**: Open `index.html` in your browser

2. **Select AI mode**: Click "Play vs AI" button

3. **Choose difficulty**: Select Easy/Medium/Hard from dropdown

4. **Play**: Make your move (you're white), AI will respond automatically

## 🔧 File Structure
```
├── index.html                    # Main game interface
├── app.js                       # Game logic with AI integration
├── pieces.js                    # Chess piece definitions
├── style.css                    # Styling
├── games.csv                    # Training dataset (20k+ games)
├── train_chess_ai_colab.py      # Google Colab training script
├── chess_api_h5.py              # API server for .h5 model
├── requirements.txt             # Python dependencies
└── SETUP_INSTRUCTIONS.md        # This file
```

## 🎮 Game Features

- **Human vs Human**: Traditional two-player mode
- **Human vs AI**: Play against trained AI opponent
- **Difficulty Levels**: Easy, Medium, Hard AI settings
- **Move History**: Tracks all moves in algebraic notation
- **AI Status**: Shows when AI is thinking and move confidence
- **Real-time**: AI responds automatically after your move

## 🧠 Model Details

- **Architecture**: Decision Transformer (attention-based)
- **Training Data**: 20,000+ historical chess games
- **Vocabulary**: ~4000 unique chess moves
- **Sequence Length**: Up to 50 moves
- **Prediction**: Uses learned patterns to suggest next moves

## 🚨 Troubleshooting

**API Server Issues**:
- Ensure both `.h5` and `.json` files are in project folder
- Check that Flask server is running on port 5000
- Verify no CORS errors in browser console

**Training Issues**:
- Make sure `games.csv` is uploaded to Colab
- Use GPU runtime in Colab for faster training
- Training takes ~10-20 minutes depending on hardware

**Game Issues**:
- Refresh browser if AI doesn't respond
- Check browser console for API connection errors
- Fallback to random moves if AI server is unavailable