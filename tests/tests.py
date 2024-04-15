import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(
#             in_features=4, out_features=2
#         )
#         self.fc2 = nn.Linear(in_features=2, out_features=1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x
#
#
# model1 = Net().to(device)
# model2 = Net().to(device)
# print([data for data in model1.parameters()])
# print([data for data in model2.parameters()])
#
#
# def copy_weights(model_new, model_old):
#     model_old.load_state_dict(model_new.state_dict())
#
#
# def equalise_policies(policy_old, policy_new):
#     """Sets the old policy's parameters equal to the new policy's parameters"""
#     # for old_param, new_param in zip(policy_old.parameters(), policy_new.parameters()):
#     #     old_param.data.copy_(new_param.data)
#     policy_old.load_state_dict(policy_new.state_dict())
#
#
# copy_weights(model1, model2)
#
# print([data for data in model1.parameters()])
# print([data for data in model2.parameters()])
#
# # model1.fc2.weight = torch.Tensor([1,2])
#
# print([data for data in model1.parameters()])
# print([data for data in model2.parameters()])
#
# print(type(model1.fc2.weight))
# print(model2.fc2.weight)
from torch.distributions import Categorical

c = torch.tensor([2, 0.5, 0.38,0.12])
d = torch.tensor([0.22, 0.5])
# s = Categorical(c)
# q = torch.randn((3, 3))
# # s.log_prob(c)
#
# print(s)
# print(s.log_prob(q))
s = c.to(d)
print(s)
