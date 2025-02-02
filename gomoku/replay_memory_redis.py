
import random
import time
import torch
import redis
import pickle
import numpy as np

class ReplayMemoryRedis:
    def __init__(self, capacity):
        self.capacity = capacity
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.key = 'replay_memory'

    def push(self, data):

        for item in data:
            board_states = item[0]
            act_probs = item[1]
            rewards = item[2]

            for i in range(len(board_states)):
                black_player_board, white_player_board, empty_board, current_player_board = board_states[i]
        
                # Make copies to handle negative strides
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

                data = pickle.dumps((states, act_prob, reward)) # serialize data
                self.redis_client.lpush(self.key, data)
        
        count = self.redis_client.llen(self.key)

        with self.redis_client.pipeline(transaction=True) as pipe:  
            if count >= self.capacity:
                pop_count = count - self.capacity
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] pop_count: {pop_count}")
                for _ in range(pop_count):
                    pipe.rpop(self.key)
                pipe.execute()

    def sample(self, batch_size):
        count = self.redis_client.llen(self.key)
        if count == 0:
            return []
        
        size = batch_size if batch_size <= count else count
        indices = random.sample(range(count), size)
        data = [self.redis_client.lindex(self.key, i) for i in indices]

        # deserialize data
        return [pickle.loads(item) for item in data]

