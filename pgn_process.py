import os
import chess
import chess.pgn
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import argparse

def parse_pgn_generator(filepath: str):
    """
    Parses a PGN file and yields games as lists of moves and average ELO ratings.
    """
    with open(filepath) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            # Get average ELO if available
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            if white_elo and black_elo:
                avg_elo = (int(white_elo) + int(black_elo)) / 2
                if avg_elo >= 2100:
                    yield list(game.mainline_moves()), avg_elo

def get_state(board: chess.Board):
    """
    Converts a chess.Board object into a NumPy array representation.
    Returns an array of shape (12, 8, 8) representing the 12 bitboards.
    """
    state = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            pieces = board.pieces(piece_type, color)
            if pieces:
                index = 6 * color + piece_type - 1
                rows, cols = divmod(np.array(list(pieces)), 8)
                state[index, rows, cols] = 1
    return state

def process_game(moves, game_index, output_path):
    """
    Processes a single game represented as a list of moves and writes training data to txt files.
    """
    board = chess.Board()

    for move_index, move in enumerate(moves):
        state = get_state(board)  # Get current board state
        board.push(move)
        next_state = get_state(board)  # Get next board state

        # Prepare file path for writing
        filename = f"game_{game_index}_move_{move_index}.txt"
        file_path = os.path.join(output_path, filename)

        # Write states to the txt file
        with open(file_path, "w") as f:
            np.savetxt(f, state.reshape(-1, 8), fmt="%.1f", header="Current State")
            f.write("\n")
            np.savetxt(f, next_state.reshape(-1, 8), fmt="%.1f", header="Next State")

    # Cleanup to free memory
    del board
    gc.collect()

def generate_training_data_from_pgn_joblib(pgn_file_path, output_path, num_workers=4):
    """
    Generates training data from a PGN file using joblib for parallel processing.
    """
    os.makedirs(output_path, exist_ok=True)
    game_generator = parse_pgn_generator(pgn_file_path)

    # Use joblib for parallel processing
    Parallel(n_jobs=num_workers)(
        delayed(process_game)(game, index, output_path) for index, (game, avg_elo) in enumerate(tqdm(game_generator, desc="Processing Games"))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data from PGN file for Chess DQN model.")
    parser.add_argument("pgn_file_path", type=str, help="Path to the input PGN file.")
    parser.add_argument("output_path", type=str, help="Directory path to store training data as txt files.")
    
    args = parser.parse_args()
    
    generate_training_data_from_pgn_joblib(args.pgn_file_path, args.output_path)