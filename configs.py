from collections import namedtuple

BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 20000
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
NUM_EPISODES = 10000

MODEL_PATH = "weights/renatus_chess_model.pth"

BOARD_SIZE = 8
SQUARE_SIZE = 64

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))