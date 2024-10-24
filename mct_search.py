import chess 
import numpy as np
from model import RenatusV2
from train import get_state

class Node:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board 
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def fully_explored_child(self):
        return len(self.children) == len(self.board.legal_moves)
    
    def expand_child(self):
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            new_board = self.board.copy()
            new_board.push(move)
            self.children.append(Node(new_board, self))
        
    def find_best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)))

def MonteSearch(root: Node, model: RenatusV2, num_iterations: int):
    for _ in range(num_iterations):
        node = root
        while node.expand_child() and node.children():
            node = node.find_best_child()
            
        if not node.fully_explored_child(): 
            node.expand_child()
            node = node.children[-1]
        
        state = get_state(node.board).unsqueeze(0)
        _, score = model(state)
        value = value.item()
        
        while node is not None:
            node.visits += 1
            node.value += value if node.board.turn == root.board.turn else -value
            node = node.parent

    return root.best_child(exploration_weight=0.0).board
        