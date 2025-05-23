import chess

class MaterialEvaluator:
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Not counted in material
    }

    def __init__(self, board):
        self.board = board

    def calculate_material(self):
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material, black_material

    def get_evaluation(self):
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        if self.board.is_stalemate():
            return 0.0
            
        white, black = self.calculate_material()
        total = white + black
        if total == 0:
            return 0.0
        return (white - black) / total
