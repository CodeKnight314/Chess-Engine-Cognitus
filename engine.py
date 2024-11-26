import chess
import chess.polyglot
import time
import random 
from score import evaluate_board
from typing import Tuple, OrderedDict
from dataclasses import dataclass
import time
from typing import Optional, Tuple, List
from tqdm import tqdm

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
    depth: int = float('inf')
    
    def __post_init__(self):
        self.unique_positions = set()
        
    def reset(self):
        self.total_nodes = 0
        self.unique_positions.clear()
        self.pruned_nodes = 0
        self.quiescence_nodes = 0
        
    def print_stats(self):
        print("\nSearch Statistics:")
        print(f"Max Depth Searched: {5 - self.depth}")
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

def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: (
        board.is_capture(move),
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
    metrics.depth = min(depth, metrics.depth)
    if depth <= 0 or board.is_game_over() or time_manager.is_time_up(): 
        return evaluate_board(board)    
    
    metrics.total_nodes += 1
    metrics.unique_positions.add(board.fen())
    
    zobrist_hash = chess.polyglot.zobrist_hash(board)
    key = (zobrist_hash, depth)
    
    if key in transposition_table:
        return transposition_table[key]
    
    moves = sorted(board.legal_moves, 
                  key=lambda m: (board.is_capture(m), board.gives_check(m)),
                  reverse=True)
    
    if board.turn: 
        max_eval = -float("inf")
        for move in moves: 
            board.push(move)
            eval = alpha_beta(depth-1, board, alpha, beta, time_manager)
            board.pop() 
            
            if eval > max_eval: 
                max_eval = eval 
                
            alpha = max(alpha, eval)
            if beta <= alpha: 
                metrics.pruned_nodes += 1
                break
        
        add_to_transposition_table(key, max_eval)
        
        return max_eval
    else: 
        min_eval = float("inf")
        for move in moves: 
            board.push(move)
            eval = alpha_beta(depth-1, board, alpha, beta, time_manager)
            board.pop()
            
            if eval < min_eval: 
                min_eval = eval
                
            beta = min(beta, eval)
            if beta <= alpha: 
                break 
            
        add_to_transposition_table(key, min_eval)

        return min_eval

def evaluate_move_single(board: chess.Board, move: chess.Move, depth: int, time_manager: TimeManager) -> Optional[Tuple[chess.Move, float]]:
    """Single-threaded move evaluation with time checking"""
    if time_manager.is_time_up():
        return None
        
    board.push(move)
    score = alpha_beta(depth, board, -float('inf'), float('inf'), time_manager)
    board.pop()
    return move, score

def find_best_move(board: chess.Board, depth: int, time_limit: float = 15.0) -> chess.Move:
    metrics.reset()
    time_manager = TimeManager(time.time(), time_limit)
    
    # Opening book check remains same
    best_move = get_opening_move(board)
    if best_move:
        return best_move
    
    moves = list(board.legal_moves)
    current_best_move = moves[0]
    current_best_score = float('-inf') if board.turn else float('inf')
    completed_moves = []
    
    for current_depth in range(1, depth + 1):
        depth_best_move = None
        depth_best_score = float('-inf') if board.turn else float('inf')
        
        print(f"\nSearching depth {current_depth}...")
        
        for move in tqdm(moves, desc=f"Evaluating {len(moves)} moves"):
            if time_manager.is_time_up():
                print("\nSearch stopped due to time limit!")
                break
                
            try:
                result = evaluate_move_single(board, move, current_depth, time_manager)
                if result is None:
                    break
                    
                evaluated_move, score = result
                
                if board.turn:
                    if score > depth_best_score:
                        depth_best_score = score
                        depth_best_move = move
                else: 
                    if score < depth_best_score:
                        depth_best_score = score
                        depth_best_move = move
                        
            except Exception as e:
                print(f"\nError evaluating move {move}: {e}")
                continue
        
        if depth_best_move is not None:
            current_best_move = depth_best_move
            current_best_score = depth_best_score
            completed_moves.append((current_best_move, current_best_score))
            print(f"Depth {current_depth} complete - Best move: {current_best_move} (score: {current_best_score:.2f})")
        
        if time_manager.is_time_up():
            break
    
    # Print final statistics
    end_time = time.time() - time_manager.start_time
    metrics.print_stats()
    print(f"\nFinal Results:")
    print(f"Time taken: {end_time:.2f} seconds")
    print(f"Maximum depth reached: {len(completed_moves)}")
    print(f"Nodes per second: {metrics.total_nodes / end_time:,.0f}")
    print(f"Best move: {current_best_move} (score: {current_best_score:.2f})")
    
    return current_best_move