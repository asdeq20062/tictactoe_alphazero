import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import BOARD_SIZE
from utils import *

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = x + y
        return self.relu(y)

class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        # initial block
        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        # middle blocks (a series of res blocks)
        self.res1 = ResBlock(128, 128)
        self.res2 = ResBlock(128, 128)
        self.res3 = ResBlock(128, 128)

        # policy head
        self.policy_conv = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(16 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # value head
        self.value_conv = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_relu = nn.ReLU()
        self.value_fc = nn.Linear(16 * BOARD_SIZE * BOARD_SIZE, 1)

    def forward(self, x):
        # initial block
        x = self.relu1(self.bn1(self.conv1(x)))

        # middle blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)   

        # value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_relu(value)
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)
        value = torch.tanh(value)

        return policy, value

class PolicyValueNetwork:
    def __init__(self, model_file=None):
        self.model = A2C()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if model_file:
            self.model.load_state_dict(torch.load(model_file, weights_only=True))
        self.model.to(self.device)

    def evaluate(self, board):
        self.model.eval()
        with torch.no_grad():
            state = convert_board_to_state_tensor(board)
            state = state.to(self.device)
            action_probs, values = self.model(state)
        return action_probs, values
    
    def get_policy_value(self, board):
        action_probs, values = self.evaluate(board)

        # Get first batch item
        action_prob = action_probs[0]
        value = values[0]

        # Convert available moves to tensor mask
        available_action_probs = []
        for move in board.get_available_moves():
            idx = move_to_index(move)
            available_action_probs.append((idx, action_prob[idx]))
        
        return available_action_probs, value
    
    def save(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def train(self, state_batch, mcts_action_probs_batch, mcts_value_batch):
        # send data to device
        state_batch = state_batch.to(self.device)
        mcts_action_probs_batch = mcts_action_probs_batch.to(self.device)
        mcts_value_batch = mcts_value_batch.to(self.device).unsqueeze(1)

        # forward
        self.model.train()
        self.optimizer.zero_grad()
        policy, value = self.model(state_batch)

        # compute loss
        value_loss = self.mse_loss(value, mcts_value_batch)
        policy_loss = self.cross_entropy_loss(policy, mcts_action_probs_batch)
        # policy_loss = -torch.mean(torch.sum(mcts_action_probs_batch * torch.log(policy), dim=1))
        loss = policy_loss + value_loss
        print(f"loss: {loss.item()}")

        # backward
        loss.backward()
        self.optimizer.step()

