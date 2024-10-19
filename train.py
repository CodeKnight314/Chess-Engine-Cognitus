import torch
import torch.nn as nn
import torch.optim as optim
import chess
import configs
import numpy as np

import argparse
from model import Renatus, choose_legal_move, ReplayMemory
from tqdm import tqdm

def get_state(board: chess.Board):
    """
    Converts a chess.Board object into a PyTorch tensor representation.
    Returns a tensor of shape (12, 8, 8) representing the 12 bitboards.
    """
    state = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            pieces = board.pieces(piece_type, color)
            for square in pieces:
                index = 6 * color + piece_type - 1
                row = square // 8
                col = square % 8
                state[index, row, col] = 1
    return torch.tensor(state, dtype=torch.float32).to(device)

def get_material_score(board: chess.Board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_score = sum(piece_values[piece] for piece in chess.PIECE_TYPES if piece in board.pieces(piece, chess.WHITE))
    black_score = sum(piece_values[piece] for piece in chess.PIECE_TYPES if piece in board.pieces(piece, chess.BLACK))
    return white_score - black_score if board.turn == chess.WHITE else black_score - white_score

def get_positional_reward(board: chess.Board):
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    reward = 0
    for square in center_squares:
        if board.piece_at(square) is not None:
            piece = board.piece_at(square)
            reward += 0.1 if piece.color == board.turn else -0.1
    return reward

def get_development_reward(board: chess.Board):
    reward = 0
    # Reward castling
    if board.is_castling(board.peek()):
        reward += 0.5
    # Reward piece development: if minor pieces (Knight/Bishop) are off their starting positions
    starting_positions = {
        chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
        chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
    }
    for square in starting_positions[board.turn]:
        if not board.piece_at(square):
            reward += 0.1
    return reward

def get_reward(board: chess.Board):
    """
    Calculates the reward for a given board state, encouraging good play.
    """
    # Winning/Draw reward
    if board.is_checkmate():
        return 1.0  # Winning
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return 0.0  # Draw

    # Material score difference reward
    material_reward = 0.01 * get_material_score(board)

    # Positional control reward
    positional_reward = get_positional_reward(board)

    # Piece development reward
    development_reward = get_development_reward(board)

    # Discourage too many moves without a goal
    negative_move_penalty = -0.01

    # Total reward calculation
    total_reward = material_reward + positional_reward + development_reward + negative_move_penalty

    return total_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Renatus(input_channels=12, num_blocks=5).to(device)
target_net = Renatus(input_channels=12, num_blocks=5).to(device)
target_net.load_state_dict(model.state_dict())
target_net.eval()

optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
memory = ReplayMemory(configs.MEMORY_SIZE)

def optimize_model():
    if len(memory) < configs.BATCH_SIZE:
        return

    transitions = memory.sample(configs.BATCH_SIZE)
    batch = configs.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    optimizer.zero_grad()
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(configs.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * configs.GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(args):
    """
    Main training loop.
    """
    steps_done = 0

    if args.path:
        pretrained_dict = torch.load(args.path, weights_only=True)
        model_dict = model.state_dict()

        # Filter out unmatched keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        target_net.load_state_dict(model_dict, strict=False)

    for episode in tqdm(range(configs.NUM_EPISODES)):
        board = chess.Board()
        state = get_state(board)
        done = False

        while not done:
            # Select and perform an action
            epsilon = configs.EPSILON_END + (configs.EPSILON_START - configs.EPSILON_END) * np.exp(-1. * steps_done / configs.EPSILON_DECAY)
            action = choose_legal_move(model, board, state, epsilon)

            # Convert action (chess.Move) to a 1D index for the network output
            action_index = torch.tensor([[action.from_square * 64 + action.to_square]], 
                                       device=device, dtype=torch.long)

            # Make the move
            next_board = board.copy()
            next_board.push(action)
            next_state = get_state(next_board) if not next_board.is_game_over() else None
            reward = torch.tensor([get_reward(next_board)], device=device)
            done = next_board.is_game_over()

            # Store the transition in memory
            memory.push(state, action_index, next_state, reward)

            # Move to the next state
            state = next_state
            board = next_board
            steps_done += 1

            # Perform one step of the optimization (on the target network)
            optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if episode % configs.TARGET_UPDATE == 0:
            target_net.load_state_dict(model.state_dict())

        # Print episode information (optional)
        if episode % 100 == 0:
            print(f"Episode {episode}/{configs.NUM_EPISODES}, Steps: {steps_done}, Epsilon: {epsilon:.4f}")

    print('Complete')
    torch.save(model.state_dict(), 'renatus_chess_model.pth')  # Save the trained model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="Pretrained model weights if available")
    args = parser.parse_args()
    train(args)