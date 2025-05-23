import pygame
import chess
from engine_transformer  import  TrainedChessEngine  # Import the trained engine class


# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 500, 500
SQ_SIZE = WIDTH // 8
PIECE_SCALE = 0.8

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (205, 210, 106)

class ChessGUI:
    def __init__(self, engine):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trained Chess AI")
        self.engine = engine
        self.flipped = True
        self.selected_square = None
        self.highlighted = []
        self.pieces = self.load_pieces()

    def load_pieces(self):
        pieces = {}
        for color in ['w', 'b']:
            for piece in ['p', 'n', 'b', 'r', 'q', 'k']:
                key = f"{color}{piece}"
                img = pygame.image.load(f'pieces/{key}.png')
                pieces[key] = pygame.transform.smoothscale(
                    img, (int(SQ_SIZE*PIECE_SCALE), int(SQ_SIZE*PIECE_SCALE)))
        return pieces

    def square_to_pos(self, square):
        """Convert chess square to screen coordinates"""
        if self.flipped:
            row = 7 - (square // 8)
            col = 7 - (square % 8)
        else:
            row = square // 8
            col = square % 8
        return (col * SQ_SIZE, row * SQ_SIZE)

    def pos_to_square(self, pos):
        """Convert screen position to chess square"""
        x, y = pos
        col = x // SQ_SIZE
        row = y // SQ_SIZE
        if self.flipped:
            col = 7 - col
            row = 7 - row
        return row * 8 + col

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row+col) % 2 == 0 else BLACK
                pygame.draw.rect(self.screen, color,
                               (col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.engine.board.piece_at(square)
            if piece:
                x, y = self.square_to_pos(square)
                piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}"
                self.screen.blit(self.pieces[piece_key],
                               (x + SQ_SIZE*(1-PIECE_SCALE)/2, 
                                y + SQ_SIZE*(1-PIECE_SCALE)/2))

    def highlight_square(self, square, color):
        x, y = self.square_to_pos(square)
        surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
        surf.set_alpha(100)
        surf.fill(color)
        self.screen.blit(surf, (x, y))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Human move handling
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.engine.board.turn == chess.WHITE:  # Human plays White
                        square = self.pos_to_square(pygame.mouse.get_pos())
                        piece = self.engine.board.piece_at(square)
                        
                        if self.selected_square is None:
                            if piece and piece.color == chess.WHITE:
                                self.selected_square = square
                                self.highlighted = [
                                    move.to_square for move in self.engine.board.legal_moves
                                    if move.from_square == square
                                ]
                        else:
                            move = chess.Move(self.selected_square, square)
                            if move in self.engine.board.legal_moves:
                                self.engine.board.push(move)
                            self.selected_square = None
                            self.highlighted = []

                # Keyboard controls
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        self.flipped = not self.flipped
                    elif event.key == pygame.K_u:
                        if len(self.engine.board.move_stack) > 0:
                            self.engine.board.pop()

            # Draw board state
            self.screen.fill((0, 0, 0))
            self.draw_board()
            
            # Highlighting
            if self.selected_square is not None:
                self.highlight_square(self.selected_square, (255, 255, 0))
                for square in self.highlighted:
                    self.highlight_square(square, HIGHLIGHT)
            
            self.draw_pieces()
            
            # AI move handling
            if not self.engine.board.is_game_over() and self.engine.board.turn == chess.BLACK:
                ai_move = self.engine.make_move()
                if ai_move:
                    print(f"AI played: {ai_move}")
                else:
                    print("AI has no valid moves!")

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    # Initialize trained engine
    trained_engine = TrainedChessEngine(model_path="trained_chess_transformer.keras")
    
    # Start GUI
    gui = ChessGUI(trained_engine)
    gui.run()