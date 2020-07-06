import numpy as np


class TabularAgent:
    def __init__(self, state_space: list, action_space: int):
        self.state_space = state_space
        self.action_space = action_space

        space = [action_space]
        space.extend(state_space)

        self.q_function = np.zeros(space)

    def e_greedy(self, state: list, e: float):
        if np.random.uniform() < e:
            return np.random.randint(self.action_space)
        else:
            return self.greedy(state)

    def greedy(self, state: list):
        index = self.build_index(state, Ellipsis)
        action = np.argmax(self.q_function[index])
        return action

    def train(self, state: list, new_state: list, action: int, reward, l_rate, gamma):
        index = self.build_index(state, action)
        index_next = self.build_index(new_state, Ellipsis)

        old_q = self.q_function[index]
        max_q = np.max(self.q_function[index_next])
        self.q_function[index] = old_q + l_rate * (reward + gamma * max_q - old_q)

    def build_index(self, state: list, action):
        index = [action]
        index.extend(state)
        return tuple(index)
