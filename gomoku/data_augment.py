
import numpy as np

from constants import *


def data_augment(data):
    # data = [(board_states, move_probs, rewards)]
    # board_states = black_player_board, white_player_board, empty_board, current_player_board
    # move_probs = [[0.1, 0.2, 0.3, 0.4]]
    # rewards = [[1, -1, 0]]

    board_shape = (BOARD_SIZE, BOARD_SIZE)
    augmented_data = []
    
    for item in data:
        board_states, move_probs, rewards = item

        augmented_board_states = []
        augmented_move_probs = []
        augmented_rewards = []

        for board_state, move_prob, reward in zip(board_states, move_probs, rewards):
            black_player_board, white_player_board, empty_board, current_player_board = board_state
            np_move_probs = np.array(move_prob)
            
            # rotate 90
            black_player_board_90 = np.rot90(black_player_board)
            white_player_board_90 = np.rot90(white_player_board)
            empty_board_90 = np.rot90(empty_board)
            current_player_board_90 = np.rot90(current_player_board)

            action_prob_90 = np_move_probs.reshape(board_shape)
            action_prob_90 = np.rot90(action_prob_90)
            action_prob_90 = action_prob_90.flatten()

            board_states_90 = [black_player_board_90, white_player_board_90, empty_board_90, current_player_board_90]

            augmented_board_states.append(board_states_90)
            augmented_move_probs.append(action_prob_90)
            augmented_rewards.append(reward)

            # rotate 180
            black_player_board_180 = np.rot90(black_player_board, 2)
            white_player_board_180 = np.rot90(white_player_board, 2)
            empty_board_180 = np.rot90(empty_board, 2)
            current_player_board_180 = np.rot90(current_player_board, 2)

            action_prob_180 = np_move_probs.reshape(board_shape)
            action_prob_180 = np.rot90(action_prob_180, 2)
            action_prob_180 = action_prob_180.flatten()

            board_states_180 = [black_player_board_180, white_player_board_180, empty_board_180, current_player_board_180]

            augmented_board_states.append(board_states_180)
            augmented_move_probs.append(action_prob_180)
            augmented_rewards.append(reward)

            # rotate 270
            black_player_board_270 = np.rot90(black_player_board, 3)
            white_player_board_270 = np.rot90(white_player_board, 3)
            empty_board_270 = np.rot90(empty_board, 3)
            current_player_board_270 = np.rot90(current_player_board, 3)

            action_prob_270 = np_move_probs.reshape(board_shape)
            action_prob_270 = np.rot90(action_prob_270, 3)
            action_prob_270 = action_prob_270.flatten()

            board_states_270 = [black_player_board_270, white_player_board_270, empty_board_270, current_player_board_270]

            augmented_board_states.append(board_states_270)
            augmented_move_probs.append(action_prob_270)
            augmented_rewards.append(reward)

        augmented_data.append((augmented_board_states, augmented_move_probs, augmented_rewards))

    return augmented_data
