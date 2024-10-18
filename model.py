import torch 
import torch.nn as nn
import random
import chess
from collections import namedtuple

class ResidualBlock(nn.Module): 
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.conv_one = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_one = nn.BatchNorm2d(output_channels)
        
        self.conv_two = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_two = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.batch_one(self.conv_one(x)))
        x = self.relu(self.batch_two(self.conv_two(x)))
        x = residual + x
        return x

class Renatus(nn.Module): 
    def __init__(self, input_channels: int, num_blocks: int):
        super().__init__()
        
        self.conv_one = nn.Conv2d(input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_two = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_three = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.res_stack = nn.Sequential(*[ResidualBlock(256, 256) for _ in range(num_blocks)])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.bottleneck = nn.Linear(16384, 1024)
        
        self.q_layers = nn.Linear(1024, 4096)
        
    def forward(self, x): 
        x = self.relu(self.conv_one(x))
        x = self.relu(self.conv_two(x))
        x = self.relu(self.conv_three(x))
        
        x = self.res_stack(x)
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        x = self.bottleneck(x)
        
        x = self.q_layers(x)
        
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def choose_legal_move(model: nn.Module, board, state, epsilon):
    with torch.no_grad():
        q_values = model(state.unsqueeze(0)).squeeze(0)
    
    q_values = q_values.reshape((64, 64))
    
    legal_moves = list(board.legal_moves)
    
    if random.random() < epsilon:
        action = random.choice(legal_moves)
    else:
        legal_q_values = torch.full((64, 64), -float('inf'))

        for move in legal_moves:
            legal_q_values[move.from_square, move.to_square] = q_values[move.from_square, move.to_square]

        best_action_index = legal_q_values.argmax()
        from_square = best_action_index // 64
        to_square = best_action_index % 64
        action = chess.Move(from_square.item(), to_square.item())

    return action
