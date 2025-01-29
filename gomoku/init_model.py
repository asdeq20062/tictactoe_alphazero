
import torch
from constants import *
from policy_value_network import PolicyValueNetwork


policy_value_network = PolicyValueNetwork(None)
torch.save(policy_value_network.model.state_dict(), NEW_MODEL_FILE)
torch.save(policy_value_network.model.state_dict(), OLD_MODEL_FILE)
