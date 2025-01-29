import random
from agent import Agent
from constants import *
from board import Board
from utils import *


class DataGenerator:
    def __init__(self, old_policy_value_network, new_policy_value_network, self_play_mode=True):
        self.old_agent = Agent(old_policy_value_network, is_self_play=self_play_mode)
        self.new_agent = Agent(new_policy_value_network, is_self_play=self_play_mode)


    def generate(self, num_games, is_print=False, thread_id="none", batch_no="none"):
        data = []

        episode_count = 0
        new_agent_win_count = 0

        for _ in range(num_games):
            board_states = []
            move_probs = []
            rewards = []
            current_players = []

            board = Board()

            if random.random() < 0.5:
                black_player = self.old_agent
                white_player = self.new_agent
                black_player_model = OLD_MODEL
                white_player_model = NEW_MODEL
            else:
                black_player = self.new_agent
                white_player = self.old_agent
                black_player_model = NEW_MODEL
                white_player_model = OLD_MODEL

            while True:
                current_player = board.get_current_player()

                if current_player == BLACK_PLAYER:
                    move_idx, move_prob = black_player.get_action(board, return_prob=True)
                    white_player.update_with_move_idx(move_idx)
                    move = index_to_move(move_idx)
                else:
                    move_idx, move_prob = white_player.get_action(board, return_prob=True)
                    black_player.update_with_move_idx(move_idx)
                    move = index_to_move(move_idx)

                # append data
                board_states.append(board.get_state())
                move_probs.append(move_prob)
                current_players.append(current_player)

                # move
                board.move(move)
                if is_print:
                    print("=========================================================")
                    board.print_board()

                # check if game is over
                winner = board.check_game_over(move)
                if winner != WINNER_NONE:
                    
                    # append rewards
                    for i in range(len(current_players)):
                        if winner == WINNER_DRAW:
                            rewards.append(-DRAW_REWARD)
                        elif winner == current_players[i]:
                            rewards.append(-WIN_REWARD)
                        else:
                            rewards.append(-LOSE_REWARD)
                    data.append((board_states, move_probs, rewards))
                    break
            
            winner_model = None
            if winner == WINNER_BLACK:
                winner_model = black_player_model
            elif winner == WINNER_WHITE:
                winner_model = white_player_model
            
            if winner_model == NEW_MODEL:
                new_agent_win_count += 1

            print(f"batch: {batch_no} - thread: {thread_id} - data generated: episode: {episode_count} - winner_model: {winner_model}")
            episode_count += 1

        return data, new_agent_win_count, episode_count

