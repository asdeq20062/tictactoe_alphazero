import random
from agent import Agent
from constants import *
from board import Board
from utils import *
import time


class DataGenerator:
    def __init__(self, old_policy_value_network, new_policy_value_network, self_play_mode=True):
        self.old_policy_value_network = old_policy_value_network
        self.new_policy_value_network = new_policy_value_network
        self.self_play_mode = self_play_mode
        self.new_agent = Agent(self.new_policy_value_network, is_self_play=self.self_play_mode)


    def generate(self, num_games, is_print=False, thread_id="none", batch_no="none"):
        data = []

        episode_count = 0

        for _ in range(num_games):
            
            board_states = []
            move_probs = []
            rewards = []
            current_players = []

            board = Board()

            while True:
                current_player = board.get_current_player()

                if current_player == BLACK_PLAYER:
                    move_idx, move_prob = self.new_agent.get_action(board, return_prob=True)
                    move = index_to_move(move_idx)
                else:
                    move_idx, move_prob = self.new_agent.get_action(board, return_prob=True)
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

                    self.new_agent.reset()
                    break

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch: {batch_no} - Thread: {thread_id} - Data generated: Episode: {episode_count} - winner: {winner} ")
            episode_count += 1

        return data

