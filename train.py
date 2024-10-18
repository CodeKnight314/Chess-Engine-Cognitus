import torch
import torch.nn as nn
import torch.optim as optim
import chess
import configs
import numpy as np

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

def get_reward(board: chess.Board):
    """
    Calculates the reward for a given board state.
    """
    if board.is_checkmate():
        return 1.0  # Winning
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return 0.0  # Draw
    else:
        return -0.01  # Small negative reward to encourage faster wins

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

def train():
    """
    Main training loop.
    """
    steps_done = 0
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
    train()