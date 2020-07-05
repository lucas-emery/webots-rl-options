import random

import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange
from typing import Optional
from controller import Node
from tabular_agent import TabularAgent
import pickle
from env_objects import Cylinder, Cube


class EnvDefs:

    epuck = ('EPUCK', 'E-puck.wbo')

    cylinders = [
        # node DEF,  node file definition, radius
        ('Cylinder1', 'Cylinder1.wbo', 0.05),
        ('Cylinder2', 'Cylinder2.wbo', 0.05)
    ]
    boxes = [
        # node DEF,  node file definition, side length
    ]


class EpuckSupervisor(SupervisorCSV):

    def __init__(self):
        super().__init__(time_step=32)
        self.observation_space = 8  # The agent has 8 inputs
        self.action_space = 4  # The agent can perform 4 actions

        self.collision_dist = 0.04  # Epuck radius = 0.037

        self.arena_size = np.array(self.supervisor.getFromDef('arena').getField('floorSize').getSFVec2f())
        self.tile_size = np.array([0.25, 0.25])
        self.tiles = np.zeros(np.ceil(self.arena_size / self.tile_size).astype(int), dtype=bool)

        self.robot: Optional[Node] = None
        self.environment_objects = []
        self.message_received = None    # Variable to save the messages received from the robot

    def reset_env(self):
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        for environment_object in self.environment_objects:
            if environment_object.webot_object:
                environment_object.webot_object.remove()

        # Respawn robot in starting position and state
        root_node = self.supervisor.getRoot()            # This gets the root of the scene tree
        children_field = root_node.getField('children')  # This gets a list of all the children, ie. objects of the scene
        epuck_def, epuck_file = EnvDefs.epuck
        children_field.importMFNode(-2, epuck_file)    # Load robot from file and add to second-to-last position
        self.robot = self.supervisor.getFromDef(epuck_def)

        self.environment_objects = self.populate_environment_objects(children_field)

    def populate_environment_objects(self, children_field):
        environment_objects = []

        for node_def, node_file, radius in EnvDefs.cylinders:
            wrapped_object = Cylinder(node_def, node_file, radius=radius)
            self.place_object_in_random_position(environment_objects, wrapped_object, children_field)
            environment_objects.append(wrapped_object)

        for node_def, node_file, side_length in EnvDefs.boxes:
            pass

        return environment_objects

    def place_object_in_random_position(self, placed_objects, wrapped_object, children_field):
        """
        Sets the shape passed by parameter to a random valid position
        within the parent's node environment.

        :param wrapped_object: the wrapped object with utility functions that is to be placed
        :param placed_objects: the objects that have already been placed
        :param children_field: the node to which the shape will be attached to
        :return: the node corresponding to the shape
        """
        children_field.importMFNode(-1, wrapped_object.node_file)
        shape = self.supervisor.getFromDef(wrapped_object.node_def)
        wrapped_object.webot_object = shape

        x, z = self.generate_random_valid_position(placed_objects, wrapped_object)

        trans_field = shape.getField('translation')
        trans_field.setSFVec3f([x, 0.05, z])
        shape.resetPhysics()

        return wrapped_object

    def generate_random_valid_position(self, placed_objects, wrapped_object):
        floor_x = 1/2
        floor_z = 1/2
        valid_position_found = False
        min_distance_from_wall = wrapped_object.get_min_distance_from_wall()
        position_x = None
        position_z = None
        while not valid_position_found:
            position_x = random.uniform(-floor_x + min_distance_from_wall, floor_x - min_distance_from_wall)
            position_z = random.uniform(-floor_z + min_distance_from_wall, floor_z - min_distance_from_wall)
            for placed_object in placed_objects:
                if placed_object.is_inside_object(position_x, position_z):
                    continue
            valid_position_found = True

        return position_x, position_z


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
        reward = -0.01

        # Reward exploration
        position = np.array([self.robot.getPosition()[0], self.robot.getPosition()[2]])
        relative_pos = position + self.arena_size/2
        tile = tuple(np.floor(relative_pos / self.tile_size).astype(int))

        if not self.tiles[tile]:
            reward += 10
            self.tiles[tile] = True

        # Punish
        # if np.any(relative_pos < self.collision_dist) or np.any(self.arena_size - relative_pos < self.collision_dist):
        #     print('collision')
        #     reward -= 1

        if self.message_received is not None:
            for i in range(8):
                if float(self.message_received[i]) > 1000:
                    # print('collision')
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


def build_state(observation, action):
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

    state.append(action)    # Last action

    return state

supervisor = EpuckSupervisor()

episode_count = 0
episode_limit = 1000
steps_per_episode = 1000
resume = False

if resume:
    with open("data.p", "rb") as f:
        data = pickle.load(f)
        agent = data['agent']
        learn_rate = data['learn_rate']
        discount_factor = data['discount_factor']
        history = data['history']
        epsilon = data['epsilon']
        epsilon_decay = data['epsilon_decay']
        print('Agent loaded. Episodes:', len(history))
else:
    agent = TabularAgent(state_space=[3, 3, 3, 3, 3, 3, 3, 3, 4], action_space=4)
    learn_rate = 1e-3
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.99
    history = []

while episode_count < episode_limit:
    episode_reward = 0
    observation = supervisor.reset()
    state = build_state(observation, np.random.randint(4))

    for step in range(steps_per_episode):
        # print('Obs', observation)
        # print('State', state)
        action = agent.e_greedy(state, e=epsilon)
        # action = agent.greedy(state)
        # print('Action', action)

        action_reward = 0
        for _ in range(4):
            new_observation, reward, done, info = supervisor.step([action])  # Action is the message sent to the robot
            action_reward += reward

        episode_reward += action_reward
        new_state = build_state(new_observation, action)

        agent.train(state, new_state, action, action_reward, learn_rate, discount_factor)

        # print('Action:', action, 'Episode reward:', episode_reward)

        if done:
            break

        observation = new_observation
        state = new_state

    episode_count += 1
    print("Episode #", episode_count, "reward:", episode_reward, "epsilon:", epsilon)
    history.append(episode_reward)
    epsilon *= epsilon_decay

    if episode_count % 100 == 0:
        with open("data.p", "wb") as f:
            pickle.dump(
                {"agent": agent, "history": history, "learn_rate": learn_rate, "discount_factor": discount_factor,
                 "epsilon": epsilon, "epsilon_decay": epsilon_decay}, f)



