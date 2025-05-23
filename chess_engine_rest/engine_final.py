import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

BOARD_SIZE = 8
CHANNELS = 13
MAX_MOVES = 4672

def board_to_tensor(board):
    piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
    tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, CHANNELS), dtype=np.float32)
    
    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8
        piece = board.piece_at(square)
        
        if piece:
            channel = piece_types.index(piece.symbol().upper()) + (6 if piece.color else 0)
            tensor[row, col, channel] = 1
        else:
            #empty square
            tensor[row, col, 12] = 1
    return tensor

def move_to_index(move):
    from_sq = move.from_square
    to_sq = move.to_square
    promotion = move.promotion or 0
    
    index = from_sq * 64 + to_sq
    if promotion:
        index += (promotion - 1) * 4096
    return index % MAX_MOVES

def create_model():
    inputs = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, CHANNELS))
    
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Policy head
    policy = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    policy = layers.Flatten()(policy)
    policy_output = layers.Dense(MAX_MOVES, activation='softmax', name='policy')(policy)
    
    # Value head
    value = layers.Conv2D(1, (1,1), activation='relu')(x)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation='relu')(value)
    value_output = layers.Dense(1, activation='tanh', name='value')(value)
    
    return models.Model(inputs, [policy_output, value_output])

class TrainedChessEngine:
    def __init__(self, model_path=None):
        if model_path:
            self.model = models.load_model(model_path)
            self._verify_model()
        else:
            self.model = create_model()
        self.board = chess.Board()
        self.temperature = 0.7  # Lower = more deterministic
        self.epsilon = 1e-10    # Numerical stability
        
    def _verify_model(self):
        """Validate model outputs"""
        test_input = np.random.rand(1, 8, 8, 13).astype(np.float32)
        policy, value = self.model.predict(test_input, verbose=0)
        
        if np.isnan(policy).any() or np.isnan(value).any():
            raise ValueError("Model outputs contain NaN values!")
            
    def set_position(self, fen):
        self.board = chess.Board(fen)
        
    def get_best_move(self):
        if self.board.is_game_over():
            return None
            
        legal_moves = list(self.board.legal_moves)
        board_tensor = board_to_tensor(self.board)[np.newaxis, ...]
        
        # Get predictions with NaN handling
        policy_pred, _ = self.model.predict(board_tensor, verbose=0)
        policy_pred = np.nan_to_num(policy_pred[0], nan=0.0)
        policy_pred = np.clip(policy_pred, self.epsilon, 1.0)
        
        legal_indices = [move_to_index(m) for m in legal_moves]
        valid_moves = [m for i, m in zip(legal_indices, legal_moves) if i < MAX_MOVES]
        valid_indices = [i for i in legal_indices if i < MAX_MOVES]
        
        if not valid_moves:
            return np.random.choice(legal_moves)
            
        legal_probs = policy_pred[valid_indices]
        
        # Add stability to probability calculations
        legal_probs += self.epsilon  # Prevent all-zero
        legal_probs = np.power(legal_probs, 1/self.temperature)
        prob_sum = np.sum(legal_probs)
        
        # Handle invalid probability sums
        if prob_sum <= self.epsilon or np.isnan(prob_sum):
            return np.random.choice(valid_moves)
            
        legal_probs /= prob_sum  # Now safe to normalize
        
        return np.random.choice(valid_moves, p=legal_probs)
    
    def make_move(self, move_uci=None):
        if move_uci:
            try:
                self.board.push_uci(move_uci)
            except:
                return False
            return True
        else:
            best_move = self.get_best_move()
            if best_move:
                self.board.push(best_move)
                return best_move.uci()
        return None

    def save_model(self, path='trained_engine.keras'):
        self.model.save(path)