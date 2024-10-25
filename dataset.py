import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import os
import chess
import chess.pgn
import chess.engine
import numpy as np
from mct_search import get_state

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

def move_to_policy(move):
    move_index = move.from_square * 64 + move.to_square
    return move_index

def get_dynamic_value_target(board, engine):
    result = engine.analyse(board, chess.engine.Limit(time=0.05))  # Adjusted time limit for efficiency
    score = result['score'].relative

    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    else:
        centipawn_value = score.score() / 100.0
        return max(-1.0, min(1.0, centipawn_value))

class PGNDataset(Dataset):
    def __init__(self, pgn_file: str, engine_path: str):
        super().__init__()
        self.pgn_file = pgn_file
        self.engine_path = engine_path
        self.games = list(parse_pgn_generator(pgn_file))
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, index):
        # Extract game information
        moves, avg_elo = self.games[index]
        board = chess.Board()

        training_data = []

        for move in moves:
            current_state = get_state(board)
            
            policy = move_to_policy(move)
            
            board.push(move)
            
            value = get_dynamic_value_target(board, self.engine)

            current_state_tensor = torch.tensor(current_state, dtype=torch.float)
            policy_tensor = torch.tensor(policy, dtype=torch.long)
            value_tensor = torch.tensor(value, dtype=torch.float)
            
            training_data.append((current_state_tensor, policy_tensor, value_tensor))

        return training_data

    def __del__(self):
        self.engine.close()

def custom_collate_fn(batch):
    current_states, policies, values = zip(*batch)
    
    # Stack the tensors to create a batch
    current_states_batch = torch.stack(current_states)
    policies_batch = torch.stack(policies)
    values_batch = torch.stack(values)
    
    return current_states_batch, policies_batch, values_batch

def get_dataloader(file_path: str, engine_path: str, batch_size: int):
    return DataLoader(PGNDataset(file_path, engine_path), batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=os.cpu_count()//2)