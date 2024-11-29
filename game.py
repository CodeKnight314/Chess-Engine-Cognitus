import pygame
import chess
from engine import find_best_move
from board import Board
from configs import *
import os
import time
import chess.pgn

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE))
        pygame.display.set_caption("Cognitus Chess")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.board_ui = Board(self.screen)
        self.selected_square = None
        self.legal_moves = []
        self.game_mode = "renatus_vs_renatus"  # or "renatus_vs_renatus"

    def run(self):
        running = True
        os.makedirs("saved_pgn/", exist_ok=True)
        filename = os.path.join("saved_pgn",f"{time.time()}-game.pgn")
        
        game = chess.pgn.Game()
        node = game
        
        if "renatus_vs_human":
            user_turn = chess.BLACK 
            comp_turn = chess.WHITE
        elif "human_vs_renatus": 
            user_turn = chess.WHITE 
            comp_turn = chess.BLACK
        else:
            user_turn = None 
            comp_turn = None
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.board.turn == user_turn:
                    usr_move = self.handle_mouse_click()
                    if usr_move:
                        node = node.add_variation(usr_move)
                        print(game, file=open(filename, "w"), end="\n\n")

            self.screen.fill((0, 0, 0))
            self.board_ui.draw(self.board, self.selected_square, self.legal_moves)
            pygame.display.flip()
            self.clock.tick(60)

            if not self.board.is_game_over():
                if self.game_mode == "renatus_vs_renatus":
                    if self.board.turn == chess.WHITE or self.board.turn == chess.BLACK:
                        pygame.time.delay(500)
                        comp_move = self.renatus_move()
                        node = node.add_variation(comp_move)
                        print(game, file=open(filename, "w"), end="\n\n")

                elif self.game_mode == "human_vs_renatus":
                    if self.board.turn == comp_turn:
                        pygame.time.delay(500)
                        comp_move = self.renatus_move()
                        node = node.add_variation(comp_move)
                        print(game, file=open(filename, "w"), end="\n\n")

                elif self.game_mode == "renatus_vs_human":
                    if self.board.turn == comp_turn:
                        pygame.time.delay(500)
                        comp_move = self.renatus_move()
                        node = node.add_variation(comp_move)
                        print(game, file=open(filename, "w"), end="\n\n")

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
                return move
            else:
                self.selected_square = None  # Deselect if invalid move
                self.legal_moves = []

    def renatus_move(self):
        """Makes a move using the Renatus model."""
        print(f"\nThinking move for {"White" if self.board.turn else "Black"}")
        best_move = find_best_move(self.board, 4, time_limit=20)
        self.board.push(best_move)
        return best_move

if __name__ == "__main__":
    game = Game()
    game.run()
