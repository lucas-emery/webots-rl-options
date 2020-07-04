import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange
from typing import Optional
from controller import Node


class EpuckSupervisor(SupervisorCSV):
    def __init__(self):
        super().__init__(time_step=256)
        self.observation_space = 8  # The agent has 4 inputs
        self.action_space = 4  # The agent can perform 2 actions

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
                observations.append(normalizeToRange(float(self.message_received[i]), 0, 4095, 0, 1.0))
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
            reward += 1
            self.tiles[tile] = True

        # Punish
        # if np.any(relative_pos < self.collision_dist) or np.any(self.arena_size - relative_pos < self.collision_dist):
        #     print('collision')
        #     reward -= 1

        print(self.message_received)
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


supervisor = EpuckSupervisor()
episode_count = 0
episode_limit = 100
steps_per_episode = 200
while episode_count < episode_limit:
    observation = supervisor.reset()
    episode_reward = 0

    for step in range(steps_per_episode):
        action = 1# np.random.randint(4)  # Agent selects action

        new_observation, reward, done, info = supervisor.step([action])  # Action is the message sent to the robot

        episode_reward += reward

        print('Action:', action, 'Episode reward:', episode_reward)

        if done:
            break

        observation = new_observation

    print("Episode #", episode_count, "reward:", episode_reward)
    episode_count += 1

