from agents.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import SubsetRandomSampler, BatchSampler
import numpy as np


class PPOAgent(Agent):
    def __init__(self, state_space: int, action_space: int, a_hidden=50, c_hidden=50,
                 a_lr=1e-3, c_lr=1e-3, gamma=0.9, clip_e=0.2):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.clip_e = clip_e

        self.actor = Actor(state_space, a_hidden, action_space)
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_space, c_hidden)
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=c_lr)

        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.state_space)  # 1st dimension is batch number
        with torch.no_grad():
            probs = self.actor(state)
        # print(probs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), probs[:, action.item()].item()

    def train(self, batch_size=8):
        # PPO algorithm
        # Unroll rewards
        rewards = np.array(self.rewards)
        reward = 0
        for i in reversed(range(len(self.rewards))):
            rewards[i] += self.gamma * reward
            reward = rewards[i]

        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long).view(-1, 1)
        old_probs = torch.tensor(self.a_probs, dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)

        for batch in BatchSampler(SubsetRandomSampler(range(len(self.states))), batch_size, drop_last=False):
            # Calculate advantage
            Rt = rewards[batch]
            V = self.critic(states[batch])
            advantage = Rt - V
            advantage.detach_()     # Inplace detach autograd

            # Calculate PPO loss
            new_probs = self.actor(states[batch]).gather(1, actions[batch])
            prob_ratio = new_probs / old_probs[batch]
            surr = prob_ratio * advantage
            clipped_surr = torch.clamp(prob_ratio, 1 - self.clip_e, 1 + self.clip_e) * advantage
            actor_loss = -torch.min(surr, clipped_surr).mean()  # Mean is here for when batch_size > 1

            # Update actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Update critic
            critic_loss = F.mse_loss(Rt, V)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self._clear_buffers()

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []


class Actor(nn.Module):
    def __init__(self, input, hidden, output):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, input, hidden):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
