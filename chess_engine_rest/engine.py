import chess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from sklearn.model_selection import train_test_split

# Constants
BOARD_SIZE = 8
CHANNELS = 13
MAX_MOVES = 4672
BATCH_SIZE = 64
EPOCHS = 20

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
    
    # Convolutional blocks
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

class ChessDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, positions, policy_targets, value_targets, batch_size):
        self.positions = positions
        self.policy_targets = policy_targets
        self.value_targets = value_targets
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.positions) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.positions[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_p = self.policy_targets[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_v = self.value_targets[idx*self.batch_size:(idx+1)*self.batch_size]
            
        return (
            tf.convert_to_tensor(batch_x, dtype=tf.float32),
            (tf.convert_to_tensor(batch_p, dtype=tf.float32), 
             tf.convert_to_tensor(batch_v, dtype=tf.float32))
        )

class ChessEngine:
    def __init__(self, model=None):
        self.model = model if model else create_model()
        self.board = chess.Board()
        
    def get_best_move(self, temperature=0.1):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
            
        board_tensor = board_to_tensor(self.board)[np.newaxis, ...]
        policy_pred, _ = self.model.predict(board_tensor, verbose=0)
        policy_pred = policy_pred[0]
        
        legal_indices = [move_to_index(m) for m in legal_moves]
        valid_moves = [m for i, m in zip(legal_indices, legal_moves) if i < MAX_MOVES]
        valid_indices = [i for i in legal_indices if i < MAX_MOVES]
        
        if not valid_moves:
            return np.random.choice(legal_moves)
            
        legal_probs = policy_pred[valid_indices]
        legal_probs = np.power(legal_probs, 1/temperature)
        legal_probs /= legal_probs.sum()
        
        return np.random.choice(valid_moves, p=legal_probs)
    
    def train(self, games_path='games.csv'):
        # Load and preprocess data
        df = pd.read_csv(games_path)
        positions = []
        policy_targets = []
        value_targets = []
        
        for _, row in df.iterrows():
            try:
                board = chess.Board()
                game_moves = row['moves'].split()
                result = 1 if row['winner'] == 'white' else -1 if row['winner'] == 'black' else 0
                
                for move_san in game_moves:
                    move = board.parse_san(move_san)
                    positions.append(board_to_tensor(board))
                    policy_target = np.zeros(MAX_MOVES)
                    policy_target[move_to_index(move)] = 1
                    policy_targets.append(policy_target)
                    value_targets.append(result if board.turn == chess.WHITE else -result)
                    board.push(move)
            except Exception as e:
                print(f"Skipping invalid game: {e}")
                continue
        
        # Split data
        X_train, X_val, y_p_train, y_p_val, y_v_train, y_v_val = train_test_split(
            positions, policy_targets, value_targets, test_size=0.1
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': losses.CategoricalCrossentropy(),
                'value': losses.MeanSquaredError()
            },
            metrics={'policy': 'accuracy', 'value': 'mae'}
        )
        
        # Create generators
        train_gen = ChessDataGenerator(X_train, y_p_train, y_v_train, BATCH_SIZE)
        val_gen = ChessDataGenerator(X_val, y_p_val, y_v_val, BATCH_SIZE)
        
        # Train
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1
        )
        
        # Save weights
        self.model.save_weights('chess_engine.weights.h5')
        return history

if __name__ == "__main__":
    # Initialize engine and train
    engine = ChessEngine()
    history = engine.train()
    
    # Save full model
    engine.model.save('trained_chess_engine.keras')

