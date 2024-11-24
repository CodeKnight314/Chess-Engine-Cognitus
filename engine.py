import chess
import chess.polyglot
from concurrent.futures import ProcessPoolExecutor
import time
import random 
from cache import PIECE_SQUARE_TABLES
from typing import Tuple, OrderedDict
from dataclasses import dataclass
import concurrent.futures
import time
import signal
from typing import Optional, Tuple, List

transposition_table = OrderedDict()
MAX_TABLE_SIZE = 1_000_000

# Timing statistics
timing_stats = {}

OPENING_BOOK_PATH = "Perfect2023.bin"

@dataclass
class SearchMetrics:
    total_nodes: int = 0
    unique_positions: set = None
    pruned_nodes: int = 0
    quiescence_nodes: int = 0
    
    def __post_init__(self):
        self.unique_positions = set()
        
    def reset(self):
        self.total_nodes = 0
        self.unique_positions.clear()
        self.pruned_nodes = 0
        self.quiescence_nodes = 0
        
    def print_stats(self):
        print("\nSearch Statistics:")
        print(f"Total nodes visited: {self.total_nodes:,}")
        print(f"Unique positions evaluated: {len(self.unique_positions):,}")
        print(f"Positions pruned: {self.pruned_nodes:,}")
        print(f"Quiescence nodes: {self.quiescence_nodes:,}")
        print(f"Node reuse rate: {((self.total_nodes - len(self.unique_positions)) / self.total_nodes * 100):.2f}%")
@dataclass
class TimeManager:
    start_time: float
    time_limit: float
    
    def is_time_up(self) -> bool:
        return time.time() - self.start_time >= self.time_limit

# Global metrics tracker
metrics = SearchMetrics()

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
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
    white_score = sum(piece_values[piece] * len(board.pieces(piece, chess.WHITE))
                      for piece in chess.PIECE_TYPES)
    black_score = sum(piece_values[piece] * len(board.pieces(piece, chess.BLACK))
                      for piece in chess.PIECE_TYPES)
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
        return 1000000 if board.turn == chess.BLACK else -1000000  # High value for winning positions
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
    w_material, w_mobility, w_king_safety, w_tactics, w_position = 2.0, 0.5, 1.0, 0.75, 0.5
    total_score = (w_material * material_score +
                   w_mobility * mobility_score +
                   w_king_safety * king_safety_score +
                   w_tactics * capture_score +
                   w_position * (positional_score + pst_score))

    return total_score

def evaluate_board(board: chess.Board):
    score = get_heuristic_score(board)
    return score

def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: (
        board.is_capture(move),
        board.gives_check(move),
        transposition_table.get((chess.polyglot.zobrist_hash(board), 0), {}).get('value', 0)
    ), reverse=True)
    return moves

def add_to_transposition_table(key: Tuple[int, int], value: float):
    """Add position to transposition table with LRU eviction"""
    if len(transposition_table) >= MAX_TABLE_SIZE:
        transposition_table.popitem(last=False)
    transposition_table[key] = value

def alpha_beta(depth: int, board: chess.Board, alpha: float, beta: float, time_manager: TimeManager) -> float:
    """
    Enhanced alpha-beta search with proper pruning
    """
    if time_manager.is_time_up():
        return board.turn and -float('inf') or float('inf')
    metrics.total_nodes += 1
    metrics.unique_positions.add(board.fen())
    
    zobrist_hash = chess.polyglot.zobrist_hash(board)
    key = (zobrist_hash, depth)

    if board.is_game_over():
        if board.is_checkmate():
            return -float('inf') if board.turn else float('inf')
        return 0

    if depth <= 0:
        return quiescence_search(board, alpha, beta)
    
    if key in transposition_table:
        return transposition_table[key]

    moves = sorted(board.legal_moves, 
                  key=lambda m: (board.is_capture(m), board.gives_check(m)),
                  reverse=True)

    if len(moves) == 0:
        return 0

    best_score = -float('inf')
    
    for move in moves:
        board.push(move)
        score = -alpha_beta(depth - 1, board, -beta, -alpha, time_manager)
        board.pop()
        
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        
        if alpha >= beta:
            metrics.pruned_nodes += 1
            break
    
    add_to_transposition_table(key, best_score)
    return best_score

def quiescence_search(board: chess.Board, alpha: float, beta: float, max_depth=2):
    """
    A refined quiescence search implementation.
    """
    metrics.total_nodes += 1
    metrics.quiescence_nodes += 1
    metrics.unique_positions.add(board.fen())
    
    stand_pat = evaluate_board(board)
    
    if stand_pat >= beta:
        metrics.pruned_nodes += 1
        return beta
    
    alpha = max(alpha, stand_pat)
    
    if max_depth is not None and max_depth <= 0:
        return stand_pat

    captures = [move for move in board.legal_moves if board.is_capture(move) or board.gives_check(move)]
    
    if not captures:
        return stand_pat

    for move in captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, None if max_depth is None else max_depth - 1)
        board.pop()
        
        if score >= beta:
            metrics.pruned_nodes += 1
            return beta
        
        alpha = max(alpha, score)
    
    return alpha


def evaluate_move_single(board: chess.Board, move: chess.Move, depth: int, time_manager: TimeManager) -> Optional[Tuple[chess.Move, float]]:
    """Single-threaded move evaluation with time checking"""
    if time_manager.is_time_up():
        return None
        
    board.push(move)
    score = -alpha_beta(depth - 1, board, -float('inf'), float('inf'), time_manager)
    board.pop()
    return move, score

def find_best_move(board: chess.Board, depth: int, time_limit: float = 15.0) -> chess.Move:
    """
    Main function to find the best move with proper time management
    Args:
        board: Current board position
        depth: Maximum search depth
        time_limit: Maximum time in seconds to search (default 15s)
    """
    # Reset metrics for new search
    metrics.reset()
    
    # Initialize time manager
    time_manager = TimeManager(time.time(), time_limit)
    
    # Check opening book
    best_move = get_opening_move(board)
    if best_move:
        return best_move
    
    moves = list(board.legal_moves)
    current_best_move = moves[0]  # Always have a move ready
    current_best_score = float('-inf') if board.turn else float('inf')
    completed_moves = []
    
    # Time allocation per move (reserve 10% for overhead)
    allocated_time_per_move = (time_limit * 0.9) / len(moves)
    
    # Evaluate moves sequentially with time checking
    for move_index, move in enumerate(moves):
        # Early exit if we're running out of time
        if time_manager.is_time_up():
            print("\nSearch stopped due to time limit!")
            break
            
        try:
            # Adjust depth based on remaining time and moves
            remaining_moves = len(moves) - move_index
            remaining_time = time_limit - (time.time() - time_manager.start_time)
            if remaining_time < 0:
                break
                
            # Dynamically adjust depth if we're running out of time
            current_depth = depth
            if remaining_time < allocated_time_per_move * remaining_moves:
                current_depth = max(1, depth - 1)  # Reduce depth if running out of time
            
            result = evaluate_move_single(board, move, current_depth, time_manager)
            if result is None:  # Time up during evaluation
                break
                
            move, score = result
            completed_moves.append((move, score))
            
            # Update best move
            if board.turn:  # White to move (maximizing)
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
            else:  # Black to move (minimizing)
                if score < current_best_score:
                    current_best_score = score
                    current_best_move = move
                    
        except Exception as e:
            print(f"\nError evaluating move {move}: {e}")
            continue
    
    end_time = time.time() - time_manager.start_time
    
    # Print search statistics
    metrics.print_stats()
    print(f"Time taken: {end_time:.2f} seconds")
    print(f"Moves evaluated: {len(completed_moves)} out of {len(moves)}")
    if end_time > 0:  # Avoid division by zero
        print(f"Nodes per second: {metrics.total_nodes / end_time:,.0f}")
    print(f"Best move: {current_best_move} (score: {current_best_score:.2f})")
    
    if end_time >= time_limit:
        print("Search terminated due to time limit!")
    
    return current_best_move