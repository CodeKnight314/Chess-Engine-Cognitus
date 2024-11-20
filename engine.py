import chess
import chess.polyglot
from concurrent.futures import ProcessPoolExecutor
import time
import random 
from piece_table import PIECE_SQUARE_TABLES

# Transposition table to store already evaluated positions
transposition_table = {}

# Timing statistics
timing_stats = {}

OPENING_BOOK_PATH = "Perfect2023.bin"

def get_opening_move(board):
    """
    Retrieves an opening move from the opening book if available.
    Selects a move probabilistically based on the frequency of moves in the book.
    """
    try:
        with chess.polyglot.open_reader(OPENING_BOOK_PATH) as reader:
            entries = list(reader.find_all(board))
            if not entries:
                return None  # No opening move available for this position

            # Create a weighted selection for probabilistic move choice
            total_weight = sum(entry.weight for entry in entries)
            weights = [entry.weight / total_weight for entry in entries]
            selected_entry = random.choices(entries, weights=weights, k=1)[0]
            return selected_entry.move
    except FileNotFoundError:
        print("Opening book file not found.")
        return None

def track_time(func_name, start_time):
    """
    Tracks and accumulates time spent on a function.
    """
    elapsed_time = time.time() - start_time
    if func_name not in timing_stats:
        timing_stats[func_name] = 0.0
    timing_stats[func_name] += elapsed_time

def get_material_score(board: chess.Board):
    start_time = time.time()
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_score = sum(piece_values[piece] * len(board.pieces(piece, chess.WHITE))
                      for piece in chess.PIECE_TYPES)
    black_score = sum(piece_values[piece] * len(board.pieces(piece, chess.BLACK))
                      for piece in chess.PIECE_TYPES)
    track_time("get_material_score", start_time)
    return white_score - black_score

def evaluate_piece_square_table(board):
    """
    Evaluates piece positions using piece-square tables.
    """
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            table = PIECE_SQUARE_TABLES.get(piece.piece_type, [0] * 64)
            if piece.color == chess.WHITE:
                score += table[square]
            else:
                score -= table[chess.square_mirror(square)]
    return score

def evaluate_captures(board):
    """
    Evaluates potential captures to favor moves with material gains.
    """
    capture_score = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            piece_captured = board.piece_at(move.to_square)
            if piece_captured:
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                capture_score += piece_values[piece_captured.piece_type]
    return capture_score

def get_checkmate_score(board):
    """
    Assigns a high score for checkmates and a penalty for stalemates.
    """
    if board.is_checkmate():
        return 10000 if board.turn == chess.BLACK else -10000  # High value for winning positions
    elif board.is_stalemate():
        return -500  # Penalty for stalemate
    return 0

def evaluate_positional_features(board):
    """
    Evaluates positional features such as center control and piece activity.
    """
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = sum(1 for square in center_squares if board.piece_at(square))
    return center_control

def get_heuristic_score(board: chess.Board):
    start_time = time.time()

    # Checkmate/Stalemate
    checkmate_score = get_checkmate_score(board)
    if checkmate_score != 0:
        return checkmate_score

    # Material, Mobility, and King Safety
    material_score = get_material_score(board)
    white_mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())
    black_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_score = white_mobility - black_mobility
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    white_king_attacks = len(board.attackers(chess.BLACK, white_king_square)) if white_king_square else 0
    black_king_attacks = len(board.attackers(chess.WHITE, black_king_square)) if black_king_square else 0
    king_safety_score = -white_king_attacks + black_king_attacks

    # Tactical and Positional Features
    capture_score = evaluate_captures(board)
    positional_score = evaluate_positional_features(board)
    pst_score = evaluate_piece_square_table(board)

    # Weights
    w_material, w_mobility, w_king_safety, w_tactics, w_position = 1.0, 0.5, 1.0, 0.75, 0.5
    total_score = (w_material * material_score +
                   w_mobility * mobility_score +
                   w_king_safety * king_safety_score +
                   w_tactics * capture_score +
                   w_position * (positional_score + pst_score))

    track_time("get_heuristic_score", start_time)
    return total_score

def evaluate_board(board: chess.Board):
    start_time = time.time()
    score = get_heuristic_score(board)
    track_time("evaluate_board", start_time)
    return score

def order_moves(board):
    start_time = time.time()
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: (
        board.is_capture(move),
        board.gives_check(move)
    ), reverse=True)
    track_time("order_moves", start_time)
    return moves

def quiescence_search(board, alpha, beta, depth=3):
    start_time = time.time()
    if depth == 0 or board.is_game_over():
        result = evaluate_board(board)
        track_time("quiescence_search", start_time)
        return result

    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        track_time("quiescence_search", start_time)
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    for move in order_moves(board):
        if not (board.is_capture(move) or board.gives_check(move)):
            continue
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, depth - 1)
        board.pop()

        if score >= beta:
            track_time("quiescence_search", start_time)
            return beta
        if score > alpha:
            alpha = score

    track_time("quiescence_search", start_time)
    return alpha

def alpha_beta(depth, board, alpha, beta):
    start_time = time.time()
    zobrist_hash = chess.polyglot.zobrist_hash(board)
    key = (zobrist_hash, depth)

    if key in transposition_table:
        track_time("alpha_beta", start_time)
        return transposition_table[key]

    if depth == 0 or board.is_game_over():
        result = quiescence_search(board, alpha, beta)
        track_time("alpha_beta", start_time)
        return result

    if board.turn == chess.WHITE:
        max_eval = -float('inf')
        for move in order_moves(board):
            board.push(move)
            eval = alpha_beta(depth - 1, board, alpha, beta)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[key] = max_eval
        track_time("alpha_beta", start_time)
        return max_eval
    else:
        min_eval = float('inf')
        for move in order_moves(board):
            board.push(move)
            eval = alpha_beta(depth - 1, board, alpha, beta)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[key] = min_eval
        track_time("alpha_beta", start_time)
        return min_eval

def evaluate_move(move, board_fen, depth):
    """
    Evaluates a single move by applying it to a copy of the board.
    """
    board = chess.Board(board_fen)
    board.push(move)
    return move, alpha_beta(depth - 1, board, -float('inf'), float('inf'))

def determine_best_move_parallel(board: chess.Board, depth: int):
    """
    Determines the best move using parallel processing at the root level.
    """
    start_time = time.time()
    moves = list(board.legal_moves)
    board_fen = board.fen()  # Use FEN to recreate the board in each process
    best_move = None
    best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            evaluate_move,
            moves,
            [board_fen] * len(moves),
            [depth] * len(moves)
        )

    for move, eval in results:
        if (board.turn == chess.WHITE and eval > best_eval) or \
           (board.turn == chess.BLACK and eval < best_eval):
            best_eval = eval
            best_move = move

    end_time = time.time()
    print(f"determine_best_move_parallel took {end_time - start_time:.4f} seconds")
    print("Timing Stats per Function:")
    for func_name, total_time in timing_stats.items():
        print(f"  {func_name}: {total_time:.4f} seconds")
    return best_move

def determine_move_with_opening_book(board, depth):
    """
    Determines the best move for the current position, prioritizing opening book moves.
    Falls back to minimax if no opening move is available.
    """
    opening_move = get_opening_move(board)
    if opening_move:
        print(f"Using opening book move: {opening_move}")
        return opening_move
    else:
        print("No opening move available. Switching to minimax.")
        return determine_best_move_parallel(board, depth)
