import copy
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, 
                model, 
                optim,
                epsilon,
                options):
        """ DQN algorithm
        
        Args:
            model (nn.Module): model defining forward network of Q function
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        super(DQN, self).__init__()
        self.model = model
        self.optim = optim
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()

        self.global_step = 0
        self.act_dim = 2
        self.gamma = options.gamma
        self.epsilon = epsilon
        self.e_decrement = (options.init_e - options.final_e)/options.exploration
        self.final_e = options.final_e
        self.actions = 2
        self.random_threshold = 0.9
        self.cuda = options.cuda

    def forward(self, obs):
        return self.model(obs)
    
    def get_action_randomly(self):
        """Get action randomly
        """
        action = np.zeros(self.actions, dtype=np.float32)
        #action_index = random.randrange(self.actions)
        action_index = 0 if random.random() < self.random_threshold else 1
        action[action_index] = 1
        return action

    def get_optim_action(self, obs):
        """Get optimal action based on current state
        """
        if self.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            q_value = self.forward(obs)
        # print(q_value)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.cpu().numpy()[0]
        action = np.zeros(self.actions, dtype=np.float32)
        action[action_index] = 1
        return action

    def get_action(self, obs):
        """Get action w.r.t current state
        """
        if self.train and random.random() <= self.epsilon:
            act = self.get_action_randomly()
        else:
            act = self.get_optim_action(obs)
        return act

    def set_train(self):
        """Set phase TRAIN
        """
        self.train = True
        self.model.train()

    def set_eval(self):
        """Set phase EVALUATION
        """
        self.train = False
        self.model.eval()

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self,
              obs,
              action,
              reward,
              next_obs,
              terminal):
        """ update value model self.model with DQN algorithm
        """
        self.optim.zero_grad()
        criterion = nn.MSELoss()

        q_value = self.model(obs)

        with torch.no_grad():
            q_value_next = self.target_model(next_obs)
            max_q, _ = torch.max(q_value_next, dim=1)
            target = reward + (1.0 - terminal) * self.gamma * max_q
            # ...

        q_value = torch.sum(torch.mul(action, q_value), dim=1)
        
        loss = criterion(q_value, target)
        # print(q_value, target, loss)
        loss.backward()
        self.optim.step()

        return loss.item()

    



# if __name__ == "__main__":
#     main()