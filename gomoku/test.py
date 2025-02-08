import torch
from board import *
from data_generator import DataGenerator
from agent import Agent
from utils import index_to_move
from policy_value_network import PolicyValueNetwork

def test_game(old_agent = None, new_agent = None):
    board = Board()
    board.print_board()

    while True:
        current_player = board.get_current_player()
        if current_player == BLACK_PLAYER:
            #move_idx = agent.get_move(board)
            move_idx = new_agent.get_action(board, return_prob=False)
            #old_agent.update_with_move_idx(move_idx)
            #move = board.get_player_move()
            #move_idx = move[0] * BOARD_SIZE + move[1]
            #move = board.get_player_move()
            #move_idx = move[0] * BOARD_SIZE + move[1]
        else:
            #move = board.get_random_move()
            #move_idx = old_agent.get_action(board, return_prob=False)
            #move_idx = new_agent.get_action(board, return_prob=False)
            #new_agent.update_with_move_idx(move_idx)
            move = board.get_player_move()
            move_idx = move[0] * BOARD_SIZE + move[1]


        move = index_to_move(move_idx)
        board.move(move)
        print("=========================================================")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] player: {current_player} - move_idx: {move_idx} - Move: {move}")
        board.print_board()
        winner = board.check_game_over(move)
        if winner != WINNER_NONE:
            print("winner: ", winner)
            break

model_file = OLD_MODEL_FILE
old_policy_value_network = PolicyValueNetwork(model_file)
old_agent = Agent(old_policy_value_network, is_self_play=False)


new_model_file = NEW_MODEL_FILE
new_policy_value_network = PolicyValueNetwork(new_model_file)
new_agent = Agent(new_policy_value_network, is_self_play=False)

test_game(old_agent, new_agent)


# old_policy_value_network = PolicyValueNetwork(OLD_MODEL_FILE)
# new_policy_value_network = PolicyValueNetwork(NEW_MODEL_FILE)


# data_generator = DataGenerator(old_policy_value_network, new_policy_value_network)
# data = data_generator.generate(1)
# print(data)
