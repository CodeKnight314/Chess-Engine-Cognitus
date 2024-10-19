import os
import chess
import chess.pgn
import numpy as np
import torch
import dill as pickle  # Use dill for robust pickling
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_pgn_generator(filepath: str):
    """
    Parses a PGN file and yields games as lists of moves.
    """
    with open(filepath) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield list(game.mainline_moves())  # Yield moves as a list to avoid recursion issues

def get_state(board: chess.Board):
    """
    Converts a chess.Board object into a PyTorch tensor representation.
    Returns a tensor of shape (12, 8, 8) representing the 12 bitboards.
    """
    state = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            pieces = board.pieces(piece_type, color)
            if pieces:
                index = 6 * color + piece_type - 1
                rows, cols = divmod(np.array(list(pieces)), 8)
                state[index, rows, cols] = 1
    return torch.from_numpy(state)

def process_game(moves):
    """
    Processes a single game represented as a list of moves and returns training data.
    """
    board = chess.Board()
    training_data = []

    for move in moves:
        state = get_state(board).numpy()  # Convert to NumPy array to avoid tensor sharing issues
        board.push(move)
        next_state = get_state(board).numpy()  # Convert to NumPy array

        training_data.append((state, next_state))

    # Cleanup to free memory
    del board
    gc.collect()
    
    return training_data

def generate_training_data_from_pgn_joblib(pgn_file_path, output_path, num_workers=4):
    """
    Generates training data from a PGN file using joblib for parallel processing.
    """
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "training_data.pkl")
    game_generator = parse_pgn_generator(pgn_file_path)

    # Use joblib for parallel processing
    results = Parallel(n_jobs=num_workers)(
        delayed(process_game)(game) for game in tqdm(game_generator, desc="Processing Games")
    )

    # Flatten the list of results into one list
    training_data = [item for sublist in results for item in sublist]

    # Write all training data to a binary file using dill
    with open(output_path, "wb") as f:
        pickle.dump(training_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data from PGN file for Chess DQN model.")
    parser.add_argument("pgn_file_path", type=str, help="Path to the input PGN file.")
    parser.add_argument("output_path", type=str, help="Path to the output pickle file to store training data.")
    
    args = parser.parse_args()
    
    generate_training_data_from_pgn_joblib(args.pgn_file_path, args.output_path)