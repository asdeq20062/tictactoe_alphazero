import gc
from multiprocessing import Pool, freeze_support
import time
import torch

from data_worker import data_worker
from game_dataset import GameDataset
from constants import *
from agent import Agent
from data_generator import DataGenerator
from data_augment import data_augment
from replay_memory_redis import ReplayMemoryRedis
from replay_memory import ReplayMemory
from torch.utils.data import DataLoader
from policy_value_network import PolicyValueNetwork

def evaluate_model():
    new_model = PolicyValueNetwork(NEW_MODEL_FILE)
    old_model = PolicyValueNetwork(OLD_MODEL_FILE)

    data_generator = DataGenerator(old_model, new_model, self_play_mode=False)
    data, new_agent_win_count, episode_count = data_generator.generate(10)

    new_agent_win_rate = new_agent_win_count / episode_count
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New agent win rate: {new_agent_win_rate:.2%}")
    return new_agent_win_rate

def main():

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
        for thread_data, new_agent_win_count, episode_count in results:
            data.extend(thread_data)
        
        data = data_augment(data)

        # save experience
        replay_memory = ReplayMemoryRedis(capacity=10000)
        replay_memory.push(data)
        
        # 训练新模型
        # use dataloader to load data
        dataset = GameDataset(data)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)\

        for i in range(5):
            policy_value_network = PolicyValueNetwork(NEW_MODEL_FILE)
            for states, act_probs, rewards in dataloader:   
                memory_batch = replay_memory.sample(400)
                for memory_item in memory_batch:

                    memory_states_batch = memory_item[0].unsqueeze(0)
                    memory_act_probs_batch = memory_item[1].unsqueeze(0)
                    memory_rewards_batch = memory_item[2].unsqueeze(0)
                    states = torch.cat([states, memory_states_batch], dim=0)
                    act_probs = torch.cat([act_probs, memory_act_probs_batch], dim=0)
                    rewards = torch.cat([rewards, memory_rewards_batch], dim=0)

                policy_value_network.train(states, act_probs, rewards)  \

        policy_value_network.save(NEW_MODEL_FILE)

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