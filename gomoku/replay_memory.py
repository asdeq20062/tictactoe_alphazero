
import random

import torch


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, data):

        count = 0
        for item in data:
            board_states = item[0]
            act_probs = item[1]
            rewards = item[2]

            for i in range(len(board_states)):
                black_player_board, white_player_board, empty_board, current_player_board = board_states[i]
        
                black_player_board_tensor = torch.tensor(black_player_board, dtype=torch.float32).unsqueeze(0)
                white_player_board_tensor = torch.tensor(white_player_board, dtype=torch.float32).unsqueeze(0) 
                empty_board_tensor = torch.tensor(empty_board, dtype=torch.float32).unsqueeze(0)
                current_player_board_tensor = torch.tensor(current_player_board, dtype=torch.float32).unsqueeze(0)

                states = torch.cat([black_player_board_tensor, white_player_board_tensor, empty_board_tensor, current_player_board_tensor], dim=0)
                act_prob = torch.tensor(act_probs[i], dtype=torch.float32)
                reward = torch.tensor(rewards[i], dtype=torch.float32)

                self.memory.append((states, act_prob, reward))
                count += 1
        
        if len(self.memory) >= self.capacity:
            pop_count = len(self.memory) - self.capacity
            for _ in range(pop_count):
                self.memory.pop(0)

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return []
        size = batch_size if batch_size <= len(self.memory) else len(self.memory)
        return random.sample(self.memory, size)
