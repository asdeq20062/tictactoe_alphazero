import gc
from multiprocessing import Pool, freeze_support
import time
import numpy as np
import torch

from data_worker import data_worker
from game_dataset import GameDataset
from constants import *
from agent import Agent
from data_generator import DataGenerator
from data_augment import data_augment
from replay_memory import ReplayMemory
from torch.utils.data import DataLoader
from policy_value_network import PolicyValueNetwork


def main():
    lr_multiplier = 1.0
    
    replay_memory = ReplayMemory(capacity=100000)

    for batch_no in range(2000):
        num_processes = 5

        pool = Pool(processes=num_processes)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current batch: {batch_no}")

        start_time = time.time()
        games_count_for_each_process = 5
        args_list = [(i, games_count_for_each_process, batch_no) for i in range(num_processes)]

        # Run processes and get results
        results = pool.map(data_worker, args_list)
        pool.close()
        pool.join()

        end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Time taken for battle: {end_time - start_time:.2f}s")
        
        # Combine all game data and win/loss statistics from threads
        data = []
        for thread_data in results:
            data.extend(thread_data)
        
        data = data_augment(data)
        
        # 训练新模型
        # use dataloader to load data
        dataset = GameDataset(data)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        policy_value_network = PolicyValueNetwork(NEW_MODEL_FILE)   

        lr_multiplier =policy_value_network.train(dataloader, replay_memory, lr_multiplier)  

        policy_value_network.save(NEW_MODEL_FILE)

        # save experience
        replay_memory.push(data)

        # replace old model with new model
        torch.save(policy_value_network.model.state_dict(), OLD_MODEL_FILE)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Old model replaced with new model")

        # 清理内存
        del policy_value_network
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()
    main()