import os
import random

import math
import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from typing import Optional
from controller import Node
from agents.tabular_agent import TabularAgent, TabularAgentMC
from agents.nn_agent import SimpleNNAgent
from agents.ppo_agent import PPOAgent
import pickle
from utils.utilities import normalize_to_range

from environment import SimpleArena, Maze
from utils.env_objects import Cylinder, Cube


class EpuckSupervisor(SupervisorCSV):
    def __init__(self):
        super().__init__(time_step=32)
        self.observation_space = 8  # The agent has 8 inputs
        self.action_space = 3  # The agent can perform 4 actions

        self.environment = Maze(self.supervisor)
        self.message_received = None    # Variable to save the messages received from the robot

    def reset_env(self):
        self.environment.reset()

    def get_observations(self):
        observations = []
        # Update self.message_received received from robot, which contains sensor data
        self.message_received = self.handle_receiver()
        if self.message_received is not None:
            for i in range(8):
                observations.append(float(self.message_received[i]))
        else:
            # Method is called before self.message_received is initialized
            for i in range(8):
                observations.append(0.0)

        return observations

    def get_reward(self, action):
        reward = 0.0

        # Reward moving forward and staying close to right wall
        if self.message_received is not None:
            if action[0] == 0:
                reward += 0.01

            if float(self.message_received[2]) > 100:      # ps2 is the sensor on the right
                reward += 0.1

            # Punish collision
            for i in range(8):
                if float(self.message_received[i]) > 1000:
                    # print('collision')
                    return -0.2

        return reward

    def is_done(self):
        return False

    def reset(self):
        self.reset_env()
        self.supervisor.simulationResetPhysics()
        self.message_received = None
        return self.get_observations()

    def get_info(self):
        return None


def build_discrete_state(observation):
    min = 100
    max = 1000
    state = []

    for i in range(8):
        value = observation[i]
        if value < min:
            state.append(0)
        elif value < max:
            state.append(1)
        else:
            state.append(2)

    return state


def build_continuous_state(observation):
    state = []

    for i in range(8):
        value = observation[i] if observation[i] > 70 else 70
        state.append(math.log(value - 40) / 3.5 - 0.97)

    return state


supervisor = EpuckSupervisor()

episode_count = 0
episode_limit = 2000
steps_per_episode = 2000
resume = False

if not os.path.exists('pickles'):
    os.makedirs('pickles')

if resume:
    with open("pickles/data.p", "rb") as f:
        data = pickle.load(f)
        agent = data['agent']
        history = data['history']
        version = data['version']
        print('Agent loaded. Version:', version, 'Episodes:', len(history))
else:
    agent = PPOAgent(state_space=supervisor.observation_space, action_space=supervisor.action_space)
    history = []

build_state = build_continuous_state

while episode_count < episode_limit:
    episode_reward = 0
    observation = supervisor.reset()
    state = build_state(observation)

    for step in range(steps_per_episode):
        action, a_prob = agent.act(state)
        #print(observation)
        #print(state)
        #print('Action', action, 'Prob', a_prob)

        action_reward = 0
        for _ in range(3):
            new_observation, reward, done, info = supervisor.step([action])  # Action is the message sent to the robot
            action_reward += reward

        episode_reward += action_reward
        new_state = build_state(new_observation)

        agent.store_transition(state, new_state, action, a_prob, action_reward)

        # print('Action:', action, 'Episode reward:', episode_reward)

        if done:
            print('Task done.')
            break

        observation = new_observation
        state = new_state

    episode_count += 1

    agent.train()

    print("Episode #", episode_count, "reward:", episode_reward)
    history.append(episode_reward)

    if episode_count % 100 == 0:
        with open("pickles/data.p", "wb") as f:
            pickle.dump({"agent": agent, "history": history, "version": '2.0'}, f)
