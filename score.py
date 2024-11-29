import chess
from cache import PIECE_SQUARE_TABLES

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def evaluate_board(board: chess.Board) -> float:
    """Evaluates chess position. Positive scores favor white, negative favor black."""
    if board.is_stalemate():
        return 0
    
    if board.is_checkmate():
        return -float('inf') if board.turn else float('inf')
        
    score = 0
    
    # Material and piece square tables
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
            
        # Material
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.BLACK:
            value = -value
            
        # Piece square tables
        table = PIECE_SQUARE_TABLES.get(piece.piece_type, [0] * 64)
        pst_value = table[square if piece.color else chess.square_mirror(square)]
        if piece.color == chess.BLACK:
            pst_value = -pst_value
            
        score += value + pst_value

    # King safety
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king and black_king:
        w_attackers = len(board.attackers(chess.BLACK, white_king))
        b_attackers = len(board.attackers(chess.WHITE, black_king))
        score += (b_attackers - w_attackers) * 10
        
    # Immediate captures bonus
    for move in board.legal_moves:
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                capture_value = PIECE_VALUES[captured_piece.piece_type]
                score += capture_value if board.turn == chess.WHITE else -capture_value

    # Mobility for both sides
    w_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.push(chess.Move.null())
    b_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()
    score += (w_moves - b_moves) * 2

    # Center control
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            score += 10 if piece.color == chess.WHITE else -10

    # Return score from proper perspective
    return score