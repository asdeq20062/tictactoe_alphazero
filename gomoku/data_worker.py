import time

import torch
from data_generator import DataGenerator
from game_dataset import GameDataset
from torch.utils.data import DataLoader
from constants import *
from replay_memory import ReplayMemory
from policy_value_network import PolicyValueNetwork



def data_worker(args):
    thread_id, num_games, batch_no, policy_value_network = args
    

    data_generator = DataGenerator(policy_value_network, policy_value_network) 
    data = data_generator.generate(num_games, is_print=False, thread_id=thread_id, batch_no=batch_no) # output: [(board_states, move_probs, rewards)]

    return data
