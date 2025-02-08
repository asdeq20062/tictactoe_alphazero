
import numpy as np
from constants import BOARD_SIZE, DIRICHLET_ALPHA
from mcts import MCTS


class Agent:
    def __init__(self, policy_value_network, is_self_play=False):
        self.mcts_player = MCTS(policy_value_network)
        self.is_self_play = is_self_play

    def reset(self):
        self.mcts_player.update_with_move_idx(-1)

    def update_with_move_idx(self, move_idx):
        self.mcts_player.update_with_move_idx(move_idx)

    def get_move(self, board):
        return self.mcts_player.get_move(board)


    def get_action(self, board, return_prob=False):
        act_idxs, probs = self.mcts_player.get_move_probs(board)
        
        move_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        move_probs[list(act_idxs)] = probs

        if self.is_self_play:
            move_idx = np.random.choice(act_idxs, p=0.75*probs + 0.25*np.random.dirichlet(DIRICHLET_ALPHA * np.ones(len(probs))))
            self.mcts_player.update_with_move_idx(move_idx)
        else:
            move_idx = np.random.choice(act_idxs, p=probs)
            move_idx = act_idxs[np.argmax(probs)] # for testing
            self.reset()

        if return_prob:
            return move_idx, move_probs
        else:
            return move_idx


