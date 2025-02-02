import numpy as np
from constants import *
from utils import *

class Node:
    def __init__(self, parent, p = 1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.q = 0
        self.p = p
        self.c_puct = C_PUCT

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
    
    def get_value(self):
        u = self.c_puct * self.p * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q + u

    def select(self):
        max_value = -float('inf')
        action_idx = None
        for key, item in self.children.items():
            value = item.get_value()
            if value > max_value:
                max_value = value
                action_idx = key
        return action_idx, self.children[action_idx]
    
    def expand(self, action_probs):
        d = np.random.dirichlet(np.ones(len(action_probs))*0.3)
        for idx, item in enumerate(action_probs):
            move_idx, prob = item
            # # add noise
            if self.parent is None:
                # add dirichlet noise
                prob =  0.75*prob + 0.25*d[idx]
            self.children[move_idx] = Node(self, prob)

    def update(self, value):
        self.visits += 1
        self.q += 1.0 * (value - self.q) / self.visits

    def backpropagate(self, value):
        if self.parent:
            self.parent.backpropagate(-value)
        self.update(value)

class MCTS:
    def __init__(self, policy_value_network):
        self.policy_value_network = policy_value_network
        self.n_playout = N_PLAYOUT
        self.root = Node(None)

    def playout(self, board):
        board = board.deepcopy()
        node = self.root
        move = None

        # select until leaf node
        while True:
            if node.is_leaf():
                break
            action, node = node.select()
            move = index_to_move(action)
            board.move(move)

        # evaluate the leaf node
        available_action_probs, leaf_value = self.policy_value_network.get_policy_value(board)
        
        # check if the game is over
        winner = board.check_game_over(move)
        if winner == WINNER_NONE:
            # expand the node
            node.expand(available_action_probs)
        else:
            if winner == WINNER_DRAW:
                leaf_value = DRAW_REWARD
            else:
                leaf_value = WIN_REWARD if winner == board.get_last_player() else LOSE_REWARD

        node.backpropagate(leaf_value)

    def get_move(self, board):
        # evaluate the leaf node
        available_action_probs, leaf_value = self.policy_value_network.get_policy_value(board)
        # return the index with the highest probability
        max_prob = -float('inf')
        best_move_idx = None
        for action_prob in available_action_probs:
            prob = action_prob[1]
            move_idx = action_prob[0]
            if prob > max_prob:
                max_prob = prob
                best_move_idx = move_idx
        return best_move_idx

    def get_move_probs(self, board):
        for i in range(self.n_playout + 1):
            board_copy = board.deepcopy()
            self.playout(board_copy)
        
        # add temperature
        act_visits = [(act_idxs, node.visits) for act_idxs, node in self.root.children.items()]
        act_idxs, visits = zip(*act_visits)

        result = np.zeros(len(act_idxs))
        for i in range(len(act_idxs)):
            result[i] = visits[i] ** (1/TEMPERATURE)

        output_result = result / np.sum(result)
        return act_idxs, output_result
        
    def update_with_move_idx(self, move_idx):
        if move_idx in self.root.children:
            self.root = self.root.children[move_idx]
            self.root.parent = None
        else:
            self.root = Node(None)

