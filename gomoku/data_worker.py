import time

import torch
from data_generator import DataGenerator
from game_dataset import GameDataset
from torch.utils.data import DataLoader
from constants import *
from replay_memory import ReplayMemory
from policy_value_network import PolicyValueNetwork



def data_worker(args):
    thread_id, num_games, batch_no = args

    old_policy_value_network = PolicyValueNetwork(OLD_MODEL_FILE)
    new_policy_value_network = PolicyValueNetwork(NEW_MODEL_FILE)
    


    data_generator = DataGenerator(old_policy_value_network, new_policy_value_network) 
    data = data_generator.generate(num_games, is_print=False, thread_id=thread_id, batch_no=batch_no) # output: [(board_states, move_probs, rewards)]

    return data
