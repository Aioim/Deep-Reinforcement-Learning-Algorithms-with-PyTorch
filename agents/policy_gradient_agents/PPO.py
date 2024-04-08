import copy
import sys
import torch
import numpy as np
from torch import optim
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.Parallel_Experience_Generator import Parallel_Experience_Generator
from utilities.Utility_Functions import normalise_rewards, create_actor_distribution


class PPO(Base_Agent):
    """Proximal Policy Optimization agent"""
    agent_name = "PPO"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy_output_size = self.calculate_policy_output_size()
        self.policy_new = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.policy_new_optimizer = optim.Adam(self.policy_new.parameters(), lr=self.hyperparameters["learning_rate"],
                                               eps=1e-4)
        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.config.seed,
                                                                  self.hyperparameters, self.action_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

    def calculate_policy_output_size(self):
        """Initialises the policies"""
        if self.action_types == "DISCRETE":  # 离散动作
            return self.action_size
        elif self.action_types == "CONTINUOUS":  # 连续动作
            return self.action_size * 2  # Because we need 1 parameter for mean and 1 for std of distribution

    def step(self):
        """Runs a step for the PPO agent"""
        # 探索策略
        exploration_epsilon = self.exploration_strategy.get_updated_epsilon_exploration(
            {"episode_number": self.episode_number})
        # 基于当前模型，多幕构造经验
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = (
            self.experience_generator.play_n_episodes(self.hyperparameters["episodes_per_learning_round"],
                                                      exploration_epsilon))
        self.episode_number += self.hyperparameters["episodes_per_learning_round"]
        # 策略模型学习
        self.policy_learn()
        # 更新模型
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.policy_new_optimizer)
        self.equalise_policies()

    def policy_learn(self):
        """A learning iteration for the policy"""
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            # 标准化数据
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        # 遍历一次学习的所有幕
        for _ in range(self.hyperparameters["learning_iterations_per_round"]):
            # 1、计算新旧模型获取各个动作的log_probability
            # 2、计算动作概率的log_probability比率差值
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            # 计算loss
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)

    def calculate_all_discounted_returns(self):
        """Calculates the cumulative discounted return for
        each episode which we will then use in a learning iteration"""
        # 计算所有幕的return_total
        all_discounted_returns = []
        for episode in range(len(self.many_episode_states)):
            discounted_returns = [0]
            # 遍历所有幕，计算每一步的return
            for ix in range(len(self.many_episode_states[episode])):
                # rewards+gamma*returns[next_states]
                return_value = (self.many_episode_rewards[episode][-(ix + 1)] +
                                self.hyperparameters["discount_rate"] * discounted_returns[-1])
                discounted_returns.append(return_value)
            # 不添加初始化的0
            discounted_returns = discounted_returns[1:]
            # 保存所有幕的所有状态的returns，从初始的state->done_state存储
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def calculate_all_ratio_of_policy_probabilities(self):
        """For each action calculates the ratio of the probability that the new policy would have picked the action vs.
         the probability the old policy would have picked it. This will then be used to inform the loss"""
        all_states = [state for states in self.many_episode_states for state in states]
        all_actions = [[action] if self.action_types == "DISCRETE" else action for actions in self.many_episode_actions
                       for action in actions]
        all_states = torch.stack([torch.Tensor(states).float().to(self.device) for states in all_states])

        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_new, all_states,
                                                                                     all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_old, all_states,
                                                                                     all_actions)
        # 新旧模型概率的log值差异
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (
                torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        # 计算动作概率的log值
        """Calculates the log probability of an action
         occuring given a policy and starting state"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        # 动作的log概率
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """Calculates the PPO loss"""
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        # torch.clamp对输入张量进行截断操作，将张量中的每个元素限制在指定的范围内。
        # all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities,
        #                                                 min=-sys.maxsize,
        #                                                 max=sys.maxsize)
        # ??? 啥子意思
        all_discounted_returns = torch.tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(
            all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        return loss

    def clamp_probability_ratio(self, value):
        # 数据截断
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return torch.clamp(input=value, min=1.0 - self.hyperparameters["clip_epsilon"],
                           max=1.0 + self.hyperparameters["clip_epsilon"])

    def take_policy_new_optimisation_step(self, loss):
        # 优化策略模型
        """Takes an optimisation step for the new policy"""
        self.policy_new_optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), self.hyperparameters[
            "gradient_clipping_norm"])  # clip gradients to help stabilise training
        self.policy_new_optimizer.step()  # this applies the gradients

    # 同步模型参数
    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)

    def save_result(self):
        """Save the results seen by the agent in the most recent experiences"""
        for ep in range(len(self.many_episode_rewards)):
            total_reward = np.sum(self.many_episode_rewards[ep])
            self.game_full_episode_scores.append(total_reward)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()
