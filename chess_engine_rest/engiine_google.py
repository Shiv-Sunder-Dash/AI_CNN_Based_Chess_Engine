import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, applications
import os

# Constants
BOARD_SIZE = 8
TARGET_SIZE = 75  # Resized input for GoogleNet
CHANNELS = 3  # 3-channel input for CNN compatibility
MAX_MOVES = 4672
BATCH_SIZE = 16
EPOCHS = 20
CHECKPOINT_PATH = "trained_engine.keras"


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

    return tf.image.resize(tensor, (TARGET_SIZE, TARGET_SIZE)).numpy().astype(np.float16)


def create_googlenet_model():
    base_model = applications.InceptionV3(include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, CHANNELS))
    x = layers.GlobalAveragePooling2D()(base_model.output)

    policy = layers.Dense(1024, activation='relu')(x)
    policy = layers.Dense(MAX_MOVES, activation='softmax', name='policy')(policy)

    value = layers.Dense(512, activation='relu')(x)
    value = layers.Dense(1, activation='tanh', name='value')(value)

    return models.Model(inputs=base_model.input, outputs=[policy, value])


class TrainedChessEngine:
    def __init__(self, model_path=None):
        self.board = chess.Board()
        if model_path and os.path.exists(model_path):
            self.model = models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = create_googlenet_model()

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
    print("Chess Engine Initialized.")
