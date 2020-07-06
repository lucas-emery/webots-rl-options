import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange
from typing import Optional
from controller import Node
from tabular_agent import TabularAgent
import pickle


class EpuckSupervisor(SupervisorCSV):
    def __init__(self):
        super().__init__(time_step=256)
        self.observation_space = 8  # The agent has 8 inputs
        self.action_space = 4  # The agent can perform 4 actions

        self.collision_dist = 0.04  # Epuck radius = 0.037

        self.arena_size = np.array(self.supervisor.getFromDef('arena').getField('floorSize').getSFVec2f())
        self.tile_size = np.array([0.25, 0.25])
        self.tiles = np.zeros(np.ceil(self.arena_size / self.tile_size).astype(int), dtype=bool)

        self.robot: Optional[Node] = None
        self.reset_env()
        self.message_received = None    # Variable to save the messages received from the robot

    def reset_env(self):
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        # Respawn robot in starting position and state
        root_node = self.supervisor.getRoot()            # This gets the root of the scene tree
        children_field = root_node.getField('children')  # This gets a list of all the children, ie. objects of the scene
        children_field.importMFNode(-2, 'E-puck.wbo')    # Load robot from file and add to second-to-last position

        # Get the new robot reference
        self.robot = self.supervisor.getFromDef('EPUCK')

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
        # Punish time
        reward = -0.1

        # Reward exploration
        position = np.array([self.robot.getPosition()[0], self.robot.getPosition()[2]])
        relative_pos = position + self.arena_size/2
        tile = tuple(np.floor(relative_pos / self.tile_size).astype(int))

        if not self.tiles[tile]:
            reward += 1
            self.tiles[tile] = True

        # Punish
        # if np.any(relative_pos < self.collision_dist) or np.any(self.arena_size - relative_pos < self.collision_dist):
        #     print('collision')
        #     reward -= 1

        if self.message_received is not None:
            for i in range(8):
                if float(self.message_received[i]) > 1000:
                    print('collision')
                    reward -= 1
                    break

        return reward

    def is_done(self):
        return False

    def reset(self):
        self.reset_env()
        self.tiles[:, :] = False
        self.supervisor.simulationResetPhysics()
        self.message_received = None
        return self.get_observations()

    def get_info(self):
        return None


def process_observation(observation):
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

supervisor = EpuckSupervisor()

episode_count = 0
episode_limit = 1000
steps_per_episode = 200
resume = False

if resume:
    with open("data.p", "rb") as f:
        data = pickle.load(f)
        agent = data['agent']
        learn_rate = data['learn_rate']
        discount_factor = data['discount_factor']
        history = data['history']
else:
    agent = TabularAgent(state_space=[3, 3, 3, 3, 3, 3, 3, 3], action_space=4)
    learn_rate = 1e-3
    discount_factor = 0.9
    history = []

while episode_count < episode_limit:
    episode_reward = 0
    observation = supervisor.reset()
    state = process_observation(observation)

    for step in range(steps_per_episode):
        # print('Obs', observation)
        # print('State', state)
        action = agent.e_greedy(state, e=0.1)

        new_observation, reward, done, info = supervisor.step([action])  # Action is the message sent to the robot
        new_state = process_observation(new_observation)
        episode_reward += reward

        agent.train(state, new_state, action, reward, learn_rate, discount_factor)

        # print('Action:', action, 'Episode reward:', episode_reward)

        if done:
            break

        observation = new_observation
        state = new_state

    print("Episode #", episode_count, "reward:", episode_reward)
    episode_count += 1
    history.append(episode_reward)

    if episode_count % 100 == 0:
        with open("data.p", "wb") as f:
            pickle.dump(
                {"agent": agent, "history": history, "learn_rate": learn_rate, "discount_factor": discount_factor}, f)



