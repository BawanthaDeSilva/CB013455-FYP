
const gameBoard = document.querySelector("#gameboard");
const playerDetails = document.querySelector("#player");
const infoDisplay = document.querySelector("#info-display");
const err = document.querySelector("#err");
const aiStatus = document.querySelector("#ai-status");
const aiModeBtn = document.querySelector("#ai-mode-btn");
const humanModeBtn = document.querySelector("#human-mode-btn");
const difficultySelect = document.querySelector("#difficulty-select");
const resetBtn = document.querySelector("#reset-btn");
const suggestMoveBtn = document.querySelector("#suggest-move-btn");
const aiThinking = document.querySelector("#ai-thinking");
const width = 8

let playerTurn = 'white';
let isAiMode = false;
let gameHistory = [];
let uciHistory = [];
let isAiThinking = false;
playerDetails.textContent = 'white'

const startPieces = [
    Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook,
    Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn,
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn,
    Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook
]

function createBoard() {
    startPieces.forEach((startPiece, i) => {
        const square = document.createElement("div");
        square.classList.add("square");
        square.innerHTML = startPiece

        square.setAttribute("square-id", i);
        square.firstChild?.setAttribute('draggable', true)

        const row = Math.floor((63 - i) / 8) + 1;

        if (row % 2 === 0) {
            square.classList.add(i % 2 == 0 ? "beige" : "brown");
        } else {
            square.classList.add(i % 2 == 0 ? "brown" : "beige");
        }

        if (i <= 15) {
            square.firstChild.firstChild.classList.add("black");
        }
        if (i >= 48) {
            square.firstChild.firstChild.classList.add("white");
        }

        gameBoard.append(square);
    });
};

createBoard();

// Initialize with White pieces highlighted after DOM is ready
setTimeout(() => {
    highlightPlayerPieces('white');
}, 100);

// Game mode event listeners
aiModeBtn.addEventListener('click', async () => {
    isAiMode = true;
    aiModeBtn.style.background = '#4CAF50';
    humanModeBtn.style.background = '#f1f1f1';
    aiStatus.textContent = '🤖 Bot Mode: You play White, Bot plays Black';
    resetGame();
});

humanModeBtn.addEventListener('click', () => {
    isAiMode = false;
    humanModeBtn.style.background = '#4CAF50';
    aiModeBtn.style.background = '#f1f1f1';
    aiStatus.textContent = '';
    resetGame();
});

resetBtn.addEventListener('click', () => {
    resetGame();
});

suggestMoveBtn.addEventListener('click', async () => {
    if (isAiThinking) return;
    
    showAiThinking();
    const suggestion = await getAiMoveSuggestion();
    hideAiThinking();
    
    if (suggestion) {
        aiStatus.textContent = `💡 Bot suggests: ${suggestion.move} (${(suggestion.confidence * 100).toFixed(1)}% confidence)`;
        highlightSuggestedMove(suggestion.move);
    }
});

difficultySelect.addEventListener('change', (e) => {
    setAiDifficulty(e.target.value);
});


const allSquares = document.querySelectorAll("#gameboard .square");
// console.log(allSquares)

allSquares.forEach(square => {
    square.addEventListener('dragstart', dragstart);
    square.addEventListener('dragover', dragover);
    square.addEventListener('drop', dragdrop);
})

let startPositionId
let draggedElement

function dragstart(e) {
    startPositionId = e.target.parentNode.getAttribute("square-id")
    draggedElement = e.target
    
    // Highlight the selected piece
    e.target.parentNode.classList.add('selected-piece');
    
    // Highlight valid moves for the selected piece
    highlightValidMoves(startPositionId, draggedElement.id);
}

function highlightValidMoves(startId, pieceType) {
    // Clear previous highlights
    clearMoveHighlights();
    
    // Highlight all valid destination squares
    for (let i = 0; i < 64; i++) {
        const targetSquare = document.querySelector(`[square-id="${i}"]`);
        if (targetSquare && isValidMoveForPiece(parseInt(startId), i, pieceType)) {
            targetSquare.classList.add('valid-move');
        }
    }
}

function clearMoveHighlights() {
    document.querySelectorAll('.valid-move').forEach(square => {
        square.classList.remove('valid-move');
    });
    document.querySelectorAll('.selected-piece').forEach(square => {
        square.classList.remove('selected-piece');
    });
}

function isValidMoveForPiece(startId, targetId, piece) {
    // Reuse the existing validation logic
    const tempDraggedElement = { 
        id: piece, 
        firstChild: { 
            classList: { 
                contains: (color) => {
                    const square = document.querySelector(`[square-id="${startId}"]`);
                    return square?.firstChild?.firstChild?.classList.contains(color);
                }
            }
        }
    };
    
    const originalDragged = draggedElement;
    const originalStartId = startPositionId;
    
    draggedElement = tempDraggedElement;
    startPositionId = startId;
    
    const targetSquare = document.querySelector(`[square-id="${targetId}"]`);
    const isValid = targetSquare ? checkIfValid(targetSquare) : false;
    
    // Restore original values
    draggedElement = originalDragged;
    startPositionId = originalStartId;
    
    return isValid;
}

function dragover(e) {
    e.preventDefault();
}

function dragdrop(e) {
    e.stopPropagation();

    const correctTurn = draggedElement.firstChild.classList.contains(playerTurn);
    const taken = e.target.classList.contains('piece');
    const valid = checkIfValid(e.target);
    const opponentTurn = playerTurn === 'white' ? 'black' : 'white';
    const takenByOpponent = e.target.firstChild?.classList.contains(opponentTurn);

    console.log('Drop info:', { correctTurn, taken, valid, takenByOpponent, playerTurn });

    if (correctTurn) {
        // Capturing opponent piece
        if (takenByOpponent && valid) {
            const targetSquare = e.target.parentNode;
            clearMoveHighlights(); // Clear highlights after move
            e.target.remove(); // Remove captured piece
            targetSquare.append(draggedElement);
            recordMove(startPositionId, targetSquare.getAttribute("square-id"), draggedElement.id).then(() => {
                checkForWin();
                changePlayer();
                // Analyze board and update suggestions after move
                setTimeout(() => analyzeAndUpdateSuggestions(), 500);
                if (isAiMode && playerTurn === 'black' && !isAiThinking) {
                    setTimeout(() => makeAiMove(), 800);
                }
            });
            return;
        }
        
        // Trying to move to occupied square (same color)
        if (taken && !takenByOpponent) {
            err.textContent = 'Cannot capture your own piece!';
            setTimeout(() => err.textContent = '', 2000);
            return;
        }
        
        // Moving to empty square
        if (valid) {
            console.log('✅ Valid move - executing');
            clearMoveHighlights(); // Clear highlights after move
            e.target.append(draggedElement);
            recordMove(startPositionId, e.target.getAttribute("square-id"), draggedElement.id).then(() => {
                checkForWin();
                changePlayer();
                // Analyze board and update suggestions after move
                setTimeout(() => analyzeAndUpdateSuggestions(), 500);
                if (isAiMode && playerTurn === 'black' && !isAiThinking) {
                    setTimeout(() => makeAiMove(), 800);
                }
            });
            return;
        }
        
        // Invalid move
        err.textContent = 'Invalid move for this piece!';
        setTimeout(() => err.textContent = '', 2000);
    } else {
        err.textContent = `Not your turn! Wait for ${playerTurn} to play.`;
        setTimeout(() => err.textContent = '', 2000);
    }
}

function checkIfValid(target) {
    const targetId = Number(target.getAttribute('square-id')) || Number(target.parentNode.getAttribute('square-id'));
    const startId = Number(startPositionId);
    const piece = draggedElement.id
    console.log(startId, targetId, piece)

    switch (piece) {
        case 'pawn':
            const isWhitePawn = draggedElement.firstChild.classList.contains('white');
            console.log('Pawn move:', { startId, targetId, isWhitePawn, playerTurn });
            
            if (isWhitePawn) {
                // White pawns start at rank 2 (squares 48-55) and move UP toward rank 8
                // In square ID system: rank 2 = squares 48-55, moving toward smaller IDs
                const starterRowWhite = [48, 49, 50, 51, 52, 53, 54, 55]; // Rank 2
                const isStartingMove = starterRowWhite.includes(startId);
                
                console.log('White pawn analysis:', { startId, targetId, diff: startId - targetId, isStartingMove });
                
                // White moves UP (decreasing square ID by 8)
                if (startId - targetId === width) {
                    console.log('✅ White pawn single move forward');
                    return true;
                }
                // Double move from starting position
                if (isStartingMove && startId - targetId === width * 2) {
                    console.log('✅ White pawn double move');
                    return true;
                }
                // Diagonal capture
                if ((startId - targetId === width - 1 || startId - targetId === width + 1) && 
                    document.querySelector(`[square-id="${targetId}"]`)?.firstChild) {
                    console.log('✅ White pawn capture');
                    return true;
                }
            } else {
                // Black pawns start at rank 7 (squares 8-15) and move DOWN toward rank 1
                // In square ID system: rank 7 = squares 8-15, moving toward larger IDs
                const starterRowBlack = [8, 9, 10, 11, 12, 13, 14, 15]; // Rank 7
                const isStartingMove = starterRowBlack.includes(startId);
                
                console.log('Black pawn analysis:', { startId, targetId, diff: targetId - startId, isStartingMove });
                
                // Black moves DOWN (increasing square ID by 8)
                if (targetId - startId === width) {
                    console.log('✅ Black pawn single move forward');
                    return true;
                }
                // Double move from starting position
                if (isStartingMove && targetId - startId === width * 2) {
                    console.log('✅ Black pawn double move');
                    return true;
                }
                // Diagonal capture
                if ((targetId - startId === width - 1 || targetId - startId === width + 1) && 
                    document.querySelector(`[square-id="${targetId}"]`)?.firstChild) {
                    console.log('✅ Black pawn capture');
                    return true;
                }
            }
            console.log('❌ Invalid pawn move');
            break;
        case 'knight':
            if (
                startId + width * 2 + 1 === targetId ||
                startId + width * 2 - 1 === targetId ||
                startId + width - 2 === targetId ||
                startId + width + 2 === targetId ||
                startId - width * 2 + 1 === targetId ||
                startId - width * 2 - 1 === targetId ||
                startId - width + 2 === targetId ||
                startId - width - 2 === targetId
            ) {
                return true
            }
            break;

        case 'bishop':
            if (
                // for right cross --- forward
                startId + width + 1 === targetId ||
                startId + width * 2 + 2 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild ||
                startId + width * 3 + 3 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild ||
                startId + width * 4 + 4 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild ||
                startId + width * 5 + 5 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild ||
                startId + width * 6 + 6 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 + 5}"]`).firstChild ||
                startId + width * 7 + 7 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 + 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 + 6}"]`).firstChild ||

                // for left cross --- forward
                startId + width - 1 === targetId ||
                startId + width * 2 - 2 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild ||
                startId + width * 3 - 3 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild ||
                startId + width * 4 - 4 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild ||
                startId + width * 5 - 5 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild ||
                startId + width * 6 - 6 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 - 5}"]`).firstChild ||
                startId + width * 7 - 7 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 - 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 - 6}"]`).firstChild ||

                // for right cross --- backward
                startId - width - 1 === targetId ||
                startId - width * 2 - 2 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild ||
                startId - width * 3 - 3 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild ||
                startId - width * 4 - 4 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild ||
                startId - width * 5 - 5 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild ||
                startId - width * 6 - 6 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 - 5}"]`).firstChild ||
                startId - width * 7 - 7 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 - 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 - 6}"]`).firstChild ||

                // fot left cross -- backward
                startId - width + 1 === targetId ||
                startId - width * 2 + 2 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild ||
                startId - width * 3 + 3 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild ||
                startId - width * 4 + 4 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild ||
                startId - width * 5 + 5 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild ||
                startId - width * 6 + 6 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 + 5}"]`).firstChild ||
                startId - width * 7 + 7 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 + 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 + 6}"]`).firstChild
            ) {
                return true;
            }
            break;

        case 'rook':
            if (
                // moving straight forward
                startId + width === targetId ||
                startId + width * 2 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild ||
                startId + width * 3 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild ||
                startId + width * 4 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild ||
                startId + width * 5 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild ||
                startId + width * 6 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 5}"]`).firstChild ||
                startId + width * 7 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 5}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 6}"]`).firstChild ||

                // moving straight backward
                startId - width === targetId ||
                startId - width * 2 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild ||
                startId - width * 3 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild ||
                startId - width * 4 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild ||
                startId - width * 5 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild ||
                startId - width * 6 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 5}"]`).firstChild ||
                startId - width * 7 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 5}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 6}"]`).firstChild ||

                // moving left side straight
                startId + 1 === targetId ||
                startId + 2 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild ||
                startId + 3 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild ||
                startId + 4 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild ||
                startId + 5 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild ||
                startId + 6 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + 5}"]`).firstChild ||
                startId + 7 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + 5}"]`).firstChild && !document.querySelector(`[square-id="${startId + 6}"]`).firstChild ||

                // moving right side straight
                startId - 1 === targetId ||
                startId - 2 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild ||
                startId - 3 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild ||
                startId - 4 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild ||
                startId - 5 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild ||
                startId - 6 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - 5}"]`).firstChild ||
                startId - 7 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - 5}"]`).firstChild && !document.querySelector(`[square-id="${startId - 6}"]`).firstChild
            ) {
                return true
            }
            break;

        case 'queen':
            if (
                // for right cross --- forward
                startId + width + 1 === targetId ||
                startId + width * 2 + 2 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild ||
                startId + width * 3 + 3 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild ||
                startId + width * 4 + 4 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild ||
                startId + width * 5 + 5 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild ||
                startId + width * 6 + 6 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 + 5}"]`).firstChild ||
                startId + width * 7 + 7 === targetId && !document.querySelector(`[square-id = "${startId + width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 + 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 + 6}"]`).firstChild ||

                // for left cross --- forward
                startId + width - 1 === targetId ||
                startId + width * 2 - 2 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild ||
                startId + width * 3 - 3 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild ||
                startId + width * 4 - 4 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild ||
                startId + width * 5 - 5 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild ||
                startId + width * 6 - 6 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 - 5}"]`).firstChild ||
                startId + width * 7 - 7 === targetId && !document.querySelector(`[square-id = "${startId + width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 5 - 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 - 6}"]`).firstChild ||

                // for right cross --- backward
                startId - width - 1 === targetId ||
                startId - width * 2 - 2 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild ||
                startId - width * 3 - 3 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild ||
                startId - width * 4 - 4 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild ||
                startId - width * 5 - 5 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild ||
                startId - width * 6 - 6 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 - 5}"]`).firstChild ||
                startId - width * 7 - 7 === targetId && !document.querySelector(`[square-id = "${startId - width - 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 - 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 - 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 - 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 - 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 - 6}"]`).firstChild ||

                // fot left cross -- backward
                startId - width + 1 === targetId ||
                startId - width * 2 + 2 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild ||
                startId - width * 3 + 3 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild ||
                startId - width * 4 + 4 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild ||
                startId - width * 5 + 5 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild ||
                startId - width * 6 + 6 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 + 5}"]`).firstChild ||
                startId - width * 7 + 7 === targetId && !document.querySelector(`[square-id = "${startId - width + 1}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 2 + 2}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 3 + 3}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 4 + 4}"]`).firstChild && !document.querySelector(`[square-id = "${startId - width * 5 + 5}"]`).firstChild && !document.querySelector(`[square-id = "${startId + width * 6 + 6}"]`).firstChild ||

                // moving straight forward
                startId + width === targetId ||
                startId + width * 2 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild ||
                startId + width * 3 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild ||
                startId + width * 4 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild ||
                startId + width * 5 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild ||
                startId + width * 6 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 5}"]`).firstChild ||
                startId + width * 7 === targetId && !document.querySelector(`[square-id="${startId + width}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 5}"]`).firstChild && !document.querySelector(`[square-id="${startId + width * 6}"]`).firstChild ||

                // moving straight backward
                startId - width === targetId ||
                startId - width * 2 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild ||
                startId - width * 3 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild ||
                startId - width * 4 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild ||
                startId - width * 5 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild ||
                startId - width * 6 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 5}"]`).firstChild ||
                startId - width * 7 === targetId && !document.querySelector(`[square-id="${startId - width}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 5}"]`).firstChild && !document.querySelector(`[square-id="${startId - width * 6}"]`).firstChild ||

                // moving left side straight
                startId + 1 === targetId ||
                startId + 2 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild ||
                startId + 3 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild ||
                startId + 4 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild ||
                startId + 5 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild ||
                startId + 6 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + 5}"]`).firstChild ||
                startId + 7 === targetId && !document.querySelector(`[square-id="${startId + 1}"]`).firstChild && !document.querySelector(`[square-id="${startId + 2}"]`).firstChild && !document.querySelector(`[square-id="${startId + 3}"]`).firstChild && !document.querySelector(`[square-id="${startId + 4}"]`).firstChild && !document.querySelector(`[square-id="${startId + 5}"]`).firstChild && !document.querySelector(`[square-id="${startId + 6}"]`).firstChild ||

                // moving right side straight
                startId - 1 === targetId ||
                startId - 2 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild ||
                startId - 3 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild ||
                startId - 4 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild ||
                startId - 5 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild ||
                startId - 6 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - 5}"]`).firstChild ||
                startId - 7 === targetId && !document.querySelector(`[square-id="${startId - 1}"]`).firstChild && !document.querySelector(`[square-id="${startId - 2}"]`).firstChild && !document.querySelector(`[square-id="${startId - 3}"]`).firstChild && !document.querySelector(`[square-id="${startId - 4}"]`).firstChild && !document.querySelector(`[square-id="${startId - 5}"]`).firstChild && !document.querySelector(`[square-id="${startId - 6}"]`).firstChild
            ) {
                return true
            }
            break;

        case 'king':
            if (
                startId + 1 === targetId ||
                startId - 1 === targetId ||
                startId + width === targetId ||
                startId + width + 1 === targetId ||
                startId + width - 1 === targetId ||
                startId - width === targetId ||
                startId - width + 1 === targetId ||
                startId - width - 1 === targetId
            ) {
                return true
            }
            break;
        default:
            break;
    }
}


function changePlayer() {
    if (playerTurn === 'white') {
        playerTurn = 'black';
        playerDetails.textContent = 'black';
        if (isAiMode) {
            aiStatus.textContent = '🤖 Bot is thinking...';
        }
        highlightPlayerPieces('black');
    } else {
        playerTurn = 'white';
        playerDetails.textContent = 'white';
        if (isAiMode) {
            aiStatus.textContent = '🎯 Your turn! Play as White';
        }
        highlightPlayerPieces('white');
    }
}

function highlightPlayerPieces(color) {
    // Remove previous highlights
    document.querySelectorAll('.current-player').forEach(el => {
        el.classList.remove('current-player');
    });
    
    // Highlight current player's pieces
    const playerPieces = document.querySelectorAll(`.${color}`);
    playerPieces.forEach(piece => {
        piece.parentNode.parentNode.classList.add('current-player');
    });
}

// Board IDs remain consistent - no flipping needed

function checkForWin() {
    const kings = Array.from(document.querySelectorAll('#king'));

    if (!kings.some(king => king.firstChild.classList.contains('white'))) {
        infoDisplay.innerHTML = "Black Player Wins!";
        const allSquares = document.querySelectorAll('.square');
        allSquares.forEach(square => square.firstChild?.setAttribute('draggable', false));
    }
    if (!kings.some(king => king.firstChild.classList.contains('black'))) {
        infoDisplay.innerHTML = "White Player Wins!";
        const allSquares = document.querySelectorAll('.square');
        allSquares.forEach(square => square.firstChild?.setAttribute('draggable', false));
    }
}

// AI and game management functions
async function recordMove(fromId, toId, piece) {
    const move = convertToAlgebraicNotation(fromId, toId, piece);
    const uciMove = convertToUCI(fromId, toId);
    gameHistory.push(move);
    uciHistory.push(uciMove);
    
    updateMoveHistory(move, uciMove);
    console.log('Move recorded:', { fromId, toId, piece, uciMove });
    
    // Sync move with backend for accurate board state
    await syncMoveWithBackend(uciMove);
}

async function syncMoveWithBackend(uciMove) {
    try {
        await fetch('http://localhost:5001/make_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: uciMove })
        });
        console.log('✅ Move synced with backend:', uciMove);
    } catch (error) {
        console.error('Failed to sync move with backend:', error);
    }
}

function updateMoveHistory(algMove, uciMove) {
    const moveList = document.querySelector('#move-list');
    const moveNumber = Math.ceil(gameHistory.length / 2);
    const isWhiteMove = (gameHistory.length % 2) === 1;
    
    if (isWhiteMove) {
        moveList.innerHTML += `<div><strong>${moveNumber}.</strong> ${algMove} (${uciMove})`;
    } else {
        const lastLine = moveList.lastElementChild;
        if (lastLine) {
            lastLine.innerHTML += ` ${algMove} (${uciMove})</div>`;
        }
    }
    
    moveList.scrollTop = moveList.scrollHeight;
}

function convertToUCI(fromId, toId) {
    const files = 'abcdefgh';
    // Standard chess coordinates: rank 1 = bottom, rank 8 = top
    const fromFile = files[fromId % 8];
    const fromRank = 8 - Math.floor(fromId / 8); // Convert square ID to rank
    const toFile = files[toId % 8];
    const toRank = 8 - Math.floor(toId / 8);
    return `${fromFile}${fromRank}${toFile}${toRank}`;
}

function convertToAlgebraicNotation(fromId, toId, piece) {
    const files = 'abcdefgh';
    const fromFile = files[fromId % 8];
    const fromRank = Math.floor(fromId / 8) + 1;
    const toFile = files[toId % 8];
    const toRank = Math.floor(toId / 8) + 1;
    
    if (piece === 'pawn') {
        return `${toFile}${toRank}`;
    } else {
        const pieceMap = {
            'king': 'K', 'queen': 'Q', 'rook': 'R',
            'bishop': 'B', 'knight': 'N'
        };
        return `${pieceMap[piece] || ''}${toFile}${toRank}`;
    }
}

async function makeAiMove() {
    if (isAiThinking) return;
    
    showAiThinking();
    
    try {
        const response = await fetch('http://localhost:5001/predict_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                moves: uciHistory,
                board_state: getCurrentBoardState()
            })
        });
        
        if (!response.ok) {
            throw new Error('AI server not available');
        }
        
        const data = await response.json();
        const aiMoveUCI = data.move;
        
        // Execute AI move with animation
        const moveExecuted = await executeAiMoveUCI(aiMoveUCI);
        
        if (moveExecuted) {
            aiStatus.textContent = `🤖 Bot played: ${aiMoveUCI} (${(data.confidence * 100).toFixed(1)}% confidence)`;
            const algMove = convertUCIToAlgebraic(aiMoveUCI);
            gameHistory.push(algMove);
            uciHistory.push(aiMoveUCI);
            updateMoveHistory(algMove, aiMoveUCI);
            
            // Sync the move with backend
            await syncMoveWithBackend(aiMoveUCI);
            
            // Wait for animation to complete before checking win/changing player
            setTimeout(() => {
                checkForWin();
                changePlayer();
                // Analyze position after AI move
                setTimeout(() => analyzeAndUpdateSuggestions(), 500);
            }, 400);
        } else {
            // Fallback to random move if UCI parsing fails
            console.log('AI move execution failed, trying random move');
            makeRandomMove();
        }
        
    } catch (error) {
        console.error('OpenAI Error:', error);
        aiStatus.textContent = '⚠️ Bot unavailable - playing random move';
        makeRandomMove();
    }
    
    hideAiThinking();
}

async function executeAiMoveUCI(uciMove) {
    try {
        // Parse UCI move (e.g., "e2e4")
        if (uciMove.length < 4) return false;
        
        const fromSquare = uciMove.substring(0, 2);
        const toSquare = uciMove.substring(2, 4);
        
        const fromId = squareToId(fromSquare);
        const toId = squareToId(toSquare);
        
        const fromElement = document.querySelector(`[square-id="${fromId}"]`);
        const toElement = document.querySelector(`[square-id="${toId}"]`);
        
        if (!fromElement || !toElement || !fromElement.firstChild) {
            return false;
        }
        
        const piece = fromElement.firstChild;
        
        // Add move animation
        piece.style.transition = 'all 0.3s ease-in-out';
        piece.style.transform = 'scale(1.1)';
        
        // Execute move after animation
        setTimeout(() => {
            if (toElement.firstChild) {
                toElement.firstChild.remove(); // Capture piece
            }
            toElement.appendChild(piece);
            piece.style.transform = 'scale(1)';
            piece.style.transition = '';
        }, 150);
        
        return true;
        
    } catch (error) {
        console.error('Error executing AI move:', error);
        return false;
    }
}

function squareToId(square) {
    // Convert chess notation (e.g., "e4") to square ID (0-63)
    // Standard chess: a1=56, h1=63, a8=0, h8=7
    const file = square.charCodeAt(0) - 97; // a=0, b=1, etc.
    const rank = parseInt(square[1]); // 1-8
    const row = 8 - rank; // Convert rank to row (rank 1 = row 7, rank 8 = row 0)
    return row * 8 + file;
}

function getCurrentBoardState() {
    // Get current board state for AI analysis
    const state = [];
    for (let i = 0; i < 64; i++) {
        const square = document.querySelector(`[square-id="${i}"]`);
        if (square && square.firstChild) {
            const piece = square.firstChild;
            const pieceType = piece.id;
            const color = piece.firstChild.classList.contains('white') ? 'w' : 'b';
            state.push(`${color}${pieceType}`);
        } else {
            state.push('');
        }
    }
    return state;
}

function convertUCIToAlgebraic(uciMove) {
    // Simple UCI to algebraic conversion
    const fromSquare = uciMove.substring(0, 2);
    const toSquare = uciMove.substring(2, 4);
    return toSquare; // Simplified
}

function makeRandomMove() {
    // Fallback random move when AI is unavailable
    const currentColorPieces = document.querySelectorAll(`.${playerTurn}`);
    if (currentColorPieces.length > 0) {
        const randomPiece = currentColorPieces[Math.floor(Math.random() * currentColorPieces.length)];
        const emptySquares = Array.from(document.querySelectorAll('.square')).filter(sq => !sq.firstChild);
        
        if (emptySquares.length > 0) {
            const randomSquare = emptySquares[Math.floor(Math.random() * emptySquares.length)];
            
            // Animate the move
            randomPiece.parentNode.style.transition = 'all 0.3s ease-in-out';
            randomPiece.parentNode.style.transform = 'scale(1.1)';
            
            setTimeout(() => {
                randomSquare.appendChild(randomPiece.parentNode);
                randomPiece.parentNode.style.transform = 'scale(1)';
                randomPiece.parentNode.style.transition = '';
                
                gameHistory.push('random');
                checkForWin();
                changePlayer();
            }, 150);
        }
    }
}

async function setAiDifficulty(difficulty) {
    try {
        await fetch('http://localhost:5001/set_difficulty', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ difficulty })
        });
    } catch (error) {
        console.error('Failed to set difficulty:', error);
    }
}

// AI Visual Feedback Functions
function showAiThinking() {
    isAiThinking = true;
    aiThinking.style.display = 'block';
    aiThinking.style.opacity = '1';
    suggestMoveBtn.disabled = true;
}

function hideAiThinking() {
    isAiThinking = false;
    aiThinking.style.display = 'none';
    suggestMoveBtn.disabled = false;
}

async function getAiMoveSuggestion() {
    try {
        const response = await fetch('http://localhost:5001/predict_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                moves: uciHistory,
                board_state: getCurrentBoardState()
            })
        });
        
        if (!response.ok) throw new Error('AI server not available');
        
        return await response.json();
    } catch (error) {
        console.error('AI suggestion error:', error);
        return null;
    }
}

async function analyzeAndUpdateSuggestions() {
    // Automatically analyze the board and provide suggestions after each move
    if (!isAiThinking) {
        console.log('🔍 Analyzing board position...');
        
        const suggestion = await getAiMoveSuggestion();
        if (suggestion) {
            const currentPlayerColor = playerTurn === 'white' ? 'White' : 'Black';
            aiStatus.textContent = `📊 Analysis: ${currentPlayerColor} should consider ${suggestion.move} (${(suggestion.confidence * 100).toFixed(1)}% confidence)`;
            
            // Highlight the suggested move briefly
            highlightSuggestedMove(suggestion.move);
        }
    }
}

function highlightSuggestedMove(uciMove) {
    // Clear previous highlights
    document.querySelectorAll('.suggested-move').forEach(el => {
        el.classList.remove('suggested-move');
    });
    
    if (uciMove.length >= 4) {
        const fromId = squareToId(uciMove.substring(0, 2));
        const toId = squareToId(uciMove.substring(2, 4));
        
        const fromSquare = document.querySelector(`[square-id="${fromId}"]`);
        const toSquare = document.querySelector(`[square-id="${toId}"]`);
        
        if (fromSquare) fromSquare.classList.add('suggested-move');
        if (toSquare) toSquare.classList.add('suggested-move');
        
        // Remove highlight after 3 seconds
        setTimeout(() => {
            if (fromSquare) fromSquare.classList.remove('suggested-move');
            if (toSquare) toSquare.classList.remove('suggested-move');
        }, 3000);
    }
}

function resetGame() {
    // Clear the board and reset to starting position
    gameBoard.innerHTML = '';
    gameHistory = [];
    uciHistory = [];
    playerTurn = 'white';  // Human always starts as White
    playerDetails.textContent = 'white';
    infoDisplay.innerHTML = '';
    err.textContent = '';
    aiStatus.textContent = isAiMode ? '🎯 Your turn! Play as White' : '';
    isAiThinking = false;
    
    // Clear move history display
    const moveList = document.querySelector('#move-list');
    if (moveList) moveList.innerHTML = '';
    
    // Reset backend board
    fetch('http://localhost:5001/reset_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }).catch(err => console.log('Backend reset failed:', err));
    
    createBoard();
    
    const allSquares = document.querySelectorAll("#gameboard .square");
    allSquares.forEach(square => {
        square.addEventListener('dragstart', dragstart);
        square.addEventListener('dragover', dragover);
        square.addEventListener('drop', dragdrop);
    });
    
    hideAiThinking();
}