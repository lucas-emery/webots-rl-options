import numpy as np

from agents.agent import Agent


class TabularAgent(Agent):
    def __init__(self, state_space: list, action_space: int, lr=1e-3, gamma=0.9, e=1, e_decay=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.e = e
        self.e_decay = e_decay

        space = [action_space]
        space.extend(state_space)

        self.q_function = np.zeros(space)

        self._clear_buffers()

    def act(self, state, policy='e_greedy'):
        if policy == 'e_greedy':
            return self._e_greedy(state)
        elif policy == 'softmax':
            return self._softmax(state)
        else:
            return self._greedy(state)

    def store_transition(self, state, new_state, action, a_prob, reward):
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.a_probs.append(a_prob)
        self.rewards.append(reward)

    def train(self, batch_size=None):
        for i in range(len(self.states)):
            index = self._build_index(self.states[i], self.actions[i])
            index_next = self._build_index(self.new_states[i], Ellipsis)

            old_q = self.q_function[index]
            max_q = np.max(self.q_function[index_next])
            self.q_function[index] = old_q + self.lr * (self.rewards[i] + self.gamma * max_q - old_q)

        self.e *= self.e_decay
        self._clear_buffers()

    def _e_greedy(self, state: list):
        if np.random.uniform() < self.e:
            return np.random.randint(self.action_space), self.e
        else:
            return self._greedy(state)

    def _greedy(self, state: list):
        index = self._build_index(state, Ellipsis)
        action = np.argmax(self.q_function[index])
        return action, 1.0

    def _softmax(self, state: list):
        index = self._build_index(state, Ellipsis)
        exps = np.exp(self.q_function[index])
        probs = exps / np.sum(exps)
        action = np.random.choice(range(self.action_space), p=probs)
        return action, probs[action]

    def _build_index(self, state: list, action):
        index = [action]
        index.extend(state)
        return tuple(index)

    def _clear_buffers(self):
        self.states = []
        self.new_states = []
        self.actions = []
        self.a_probs = []
        self.rewards = []


class TabularAgentMC(TabularAgent):
    def train(self, batch_size=None):
        discounted_reward = 0
        for i in reversed(range(len(self.states))):
            index = self._build_index(self.states[i], self.actions[i])

            old_q = self.q_function[index]
            discounted_reward = self.rewards[i] + self.gamma * discounted_reward
            self.q_function[index] = old_q + self.lr * (discounted_reward - old_q)

        self.e *= self.e_decay
        self._clear_buffers()
