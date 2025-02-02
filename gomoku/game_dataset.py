import numpy as np
import torch
from torch.utils.data import Dataset

class GameDataset(Dataset):
    def __init__(self, data):
        self.states_batch = []
        self.act_probs_batch = []
        self.rewards_batch = []

        for item in data:
            board_states = item[0]
            act_probs = item[1]
            rewards = item[2]

            for i in range(len(board_states)):
                black_player_board, white_player_board, empty_board, current_player_board = board_states[i]


                black_player_board = np.array(black_player_board).copy()
                white_player_board = np.array(white_player_board).copy()
                empty_board = np.array(empty_board).copy()
                current_player_board = np.array(current_player_board).copy()

                black_player_board_tensor = torch.tensor(black_player_board, dtype=torch.float32).unsqueeze(0)
                white_player_board_tensor = torch.tensor(white_player_board, dtype=torch.float32).unsqueeze(0) 
                empty_board_tensor = torch.tensor(empty_board, dtype=torch.float32).unsqueeze(0)
                current_player_board_tensor = torch.tensor(current_player_board, dtype=torch.float32).unsqueeze(0)

                states = torch.cat([black_player_board_tensor, white_player_board_tensor, empty_board_tensor, current_player_board_tensor], dim=0)
                act_prob = torch.tensor(act_probs[i], dtype=torch.float32)
                reward = torch.tensor(rewards[i], dtype=torch.float32)

                self.states_batch.append(states)
                self.act_probs_batch.append(act_prob)
                self.rewards_batch.append(reward)

    def __len__(self):
        return len(self.states_batch)

    def __getitem__(self, idx):
        return self.states_batch[idx], self.act_probs_batch[idx], self.rewards_batch[idx]

