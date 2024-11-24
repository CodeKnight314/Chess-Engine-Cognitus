import pygame
import chess
from model import RenatusV1  # Import from your model.py
from engine import find_best_move
from board import Board
from configs import *  # Import constants from configs.py
import torch
from train import get_state
import os

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE))
        pygame.display.set_caption("Renatus Chess")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.board_ui = Board(self.screen)
        self.selected_square = None
        self.legal_moves = []
        self.game_mode = "renatus_vs_renatus"  # or "renatus_vs_renatus"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.renatus_model = RenatusV1(input_channels=12, num_blocks=5).to(self.device)
        if os.path.exists(MODEL_PATH):
            self.renatus_model.load_state_dict(torch.load(MODEL_PATH))  # Load your trained model
        self.renatus_model.eval()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.board.turn == chess.WHITE:
                    # Only handle mouse clicks when it's human's turn in human_vs_renatus
                    self.handle_mouse_click()

            # Drawing board and pieces
            self.screen.fill((0, 0, 0))  # Clear the screen
            self.board_ui.draw(self.board, self.selected_square, self.legal_moves)
            pygame.display.flip()
            self.clock.tick(60)  # Limit frame rate

            # Handle Renatus move when it is Renatus's turn
            if self.game_mode == "renatus_vs_renatus" and not self.board.is_game_over():
                if self.board.turn == chess.WHITE or self.board.turn == chess.BLACK:
                    pygame.time.delay(500)  # Delay to make moves readable
                    self.renatus_move()
            elif self.game_mode == "human_vs_renatus" and self.board.turn == chess.BLACK and not self.board.is_game_over():
                pygame.time.delay(500)  # Delay to make Renatus move visually readable
                self.renatus_move()

        pygame.quit()

    def handle_mouse_click(self):
        """Handles mouse clicks to select and move pieces."""
        x, y = pygame.mouse.get_pos()
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        clicked_square = chess.square(col, 7 - row)

        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_moves = [move for move in self.board.legal_moves if move.from_square == self.selected_square]
        else:
            # Make a move
            move = chess.Move(self.selected_square, clicked_square)
            if move in self.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
            else:
                self.selected_square = None  # Deselect if invalid move
                self.legal_moves = []

    def renatus_move(self):
        """Makes a move using the Renatus model."""
        print(f"\nThinking move for {"white" if self.board.turn else "Black"}")
        best_move = find_best_move(self.board, 4)
        self.board.push(best_move)

if __name__ == "__main__":
    game = Game()
    game.run()
