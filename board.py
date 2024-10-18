import pygame
import chess

class Board:
    def __init__(self, screen):
        self.screen = screen
        self.board_size = 8  # Number of squares on each side
        self.square_size = 64  # Size of each square in pixels
        self.board_color1 = (240, 217, 181)  # Light square color
        self.board_color2 = (181, 136, 99)  # Dark square color
        self.highlight_color = (200, 200, 0)  # Color for highlighting squares
        self.piece_images = {}  # Dictionary to store piece images

        self.load_piece_images()

    def load_piece_images(self):
        """Loads images for the chess pieces."""
        white_piece_names = ['P', 'N', 'B', 'R', 'Q', 'K']
        black_piece_names = ['p', 'n', 'b', 'r', 'q', 'k']
        for piece_name in white_piece_names:
            image_path = f"images/White/{piece_name}.png"  # Assuming images are in an 'images' folder
            self.piece_images[piece_name] = pygame.image.load(image_path)
        for piece_name in black_piece_names: 
            image_path = f"images/Black/{piece_name}.png"
            self.piece_images[piece_name] = pygame.image.load(image_path)

    def draw(self, board, selected_square=None, legal_moves=[]):
        """Draws the chessboard and pieces."""
        for row in range(self.board_size):
            for col in range(self.board_size):
                square_x = col * self.square_size
                square_y = row * self.square_size
                color = self.board_color1 if (row + col) % 2 == 0 else self.board_color2
                pygame.draw.rect(self.screen, color, (square_x, square_y, self.square_size, self.square_size))

                piece = board.piece_at(chess.square(col, 7 - row))  # Note: Flipping row for correct orientation
                if piece:
                    piece_name = piece.symbol()
                    image = self.piece_images[piece_name]
                    image_rect = image.get_rect(center=(square_x + self.square_size // 2, square_y + self.square_size // 2))
                    self.screen.blit(image, image_rect)

                # Highlight selected square and legal moves
                square_index = row * 8 + col
                if square_index == selected_square:
                    pygame.draw.rect(self.screen, self.highlight_color, (square_x, square_y, self.square_size, self.square_size), 4)
                if chess.Move(selected_square, square_index) in legal_moves:
                    pygame.draw.circle(self.screen, self.highlight_color, (square_x + self.square_size // 2, square_y + self.square_size // 2), 10)
                    
def test_board_visualization():
    """
    Tests the visualization of the chessboard using Pygame.
    """
    pygame.init()
    screen = pygame.display.set_mode((8 * 64, 8 * 64))  # 8x8 board with 64x64 pixel squares
    pygame.display.set_caption("Chessboard Test")
    board = chess.Board()
    board_ui = Board(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear the screen   

        board_ui.draw(board)  # Draw the chessboard
        pygame.display.flip()  # Update the display

    pygame.quit()
    
if __name__ == "__main__":
    test_board_visualization()