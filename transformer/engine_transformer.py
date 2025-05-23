import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import os

# Constants
BOARD_SIZE = 8
CHANNELS = 3  # 3-channel input for Transformer compatibility
MAX_MOVES = 4672
BATCH_SIZE = 16
EPOCHS = 20
CHECKPOINT_PATH = "trained_chess_transformer.keras"


def board_to_tensor(board):
    piece_values = {
        'P': [1, 0, 0], 'N': [0, 1, 0], 'B': [0, 0, 1], 'R': [1, 1, 0],
        'Q': [1, 0, 1], 'K': [0, 1, 1]
    }
    tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, CHANNELS), dtype=np.float32)

    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8
        piece = board.piece_at(square)

        if piece:
            color_factor = 1 if piece.color == chess.WHITE else 0.5
            tensor[row, col] = np.array(piece_values.get(piece.symbol().upper(), [0, 0, 0])) * color_factor

    return tensor.astype(np.float16)


def create_transformer_model():
    inputs = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, CHANNELS))
    x = layers.Reshape((BOARD_SIZE * BOARD_SIZE, CHANNELS))(inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    x = layers.Flatten()(x)

    policy = layers.Dense(MAX_MOVES, activation='softmax', name='policy')(x)
    value = layers.Dense(1, activation='tanh', name='value')(x)

    return models.Model(inputs=inputs, outputs=[policy, value])


class TrainedChessEngine:
    def __init__(self, model_path=None):
        self.board = chess.Board()
        if model_path and os.path.exists(model_path):
            self.model = models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = create_transformer_model()

    def make_move(self):
        if self.board.is_game_over():
            return None

        board_tensor = np.expand_dims(board_to_tensor(self.board), axis=0)
        policy, _ = self.model.predict(board_tensor, verbose=0)

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        best_move = max(legal_moves, key=lambda move: policy[0][move.from_square * 64 + move.to_square])
        self.board.push(best_move)
        return best_move

    def reset_board(self):
        self.board.reset()


if __name__ == "__main__":
    engine = TrainedChessEngine(model_path=CHECKPOINT_PATH)
    print("Transformer Chess Engine Initialized.")


