import tensorflow as tf  # Added missing import
from engine_final import TrainedChessEngine

def main():
    # Initialize engine with trained model
    engine = TrainedChessEngine(model_path='trained_chess_engine.keras')
    
    # Test position (start position)
    engine.set_position('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    
    # Get AI move
    ai_move = engine.make_move()
    print(f"AI played: {ai_move}")
    
    # Display board
    print("\nCurrent board:")
    print(engine.board.unicode())

if __name__ == "__main__":
    # Enable GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    main()
