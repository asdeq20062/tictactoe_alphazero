
import torch

from constants import BOARD_SIZE

# output is (batch_size, 4, board_size, board_size)
def convert_board_to_state_tensor(board, device):
    black_player_board, white_player_board, empty_board, current_player_board = board.get_state()
    

    black_player_board_tensor = torch.tensor(black_player_board, dtype=torch.float32, device=device).unsqueeze(0)
    white_player_board_tensor = torch.tensor(white_player_board, dtype=torch.float32, device=device).unsqueeze(0) 
    empty_board_tensor = torch.tensor(empty_board, dtype=torch.float32, device=device).unsqueeze(0)
    current_player_board_tensor = torch.tensor(current_player_board, dtype=torch.float32, device=device).unsqueeze(0)
    
    return torch.cat([black_player_board_tensor, white_player_board_tensor, empty_board_tensor, current_player_board_tensor], dim=0).unsqueeze(0)

def move_to_index(move):
    return move[0] * BOARD_SIZE + move[1]

def index_to_move(index):
    return (index // BOARD_SIZE, index % BOARD_SIZE)
