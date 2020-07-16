import math

import numpy as np
from utils.env_objects import Cylinder, Cube
import os


class EnvDefs:
    epuck = ('EPUCK', os.path.abspath('webots_objects/E-puck.wbo'))
    cylinders = [
        # node DEF,  node file definition, radius
        ('Cylinder1', os.path.abspath('webots_objects/Cylinder1.wbo'), 0.05),
        ('Cylinder2', os.path.abspath('webots_objects/Cylinder2.wbo'), 0.05)
    ]
    boxes = [
        # node DEF,  node file definition, side length
        ('Box1', os.path.abspath('webots_objects/Box1.wbo'), 0.1),
        ('Box2', os.path.abspath('webots_objects/Box2.wbo'), 0.1)
    ]
    wall = ('Wall', os.path.abspath('webots_objects/Wall.wbo'))


class SimpleArena:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        # initialization helper variables
        self.robot_initial_position = []
        self.children_field = self.supervisor.getRoot().getField('children')  # Get list of all the objects of the scene
        self.robot = None
        self.environment_objects = []

    def reset(self):
        self._remove_objects()

        # Respawn robot in starting position and state
        epuck_def, epuck_file = EnvDefs.epuck
        self.children_field.importMFNode(-2, epuck_file)  # Load robot from file and add to second-to-last position
        self.robot = self.supervisor.getFromDef(epuck_def)

        self._insert_robot_in_random_position()
        self.environment_objects = self._populate_environment_objects()

    def _remove_objects(self):
        if self.robot is not None:
            self.robot.remove()

        for environment_object in self.environment_objects:
            if environment_object.webot_object:
                environment_object.webot_object.remove()

    def get_robot(self):
        return self.robot

    def _populate_environment_objects(self):
        environment_objects = []

        for node_def, node_file, radius in EnvDefs.cylinders:
            wrapped_object = Cylinder(node_def, node_file, radius=radius)
            self._place_object_in_random_position(environment_objects, wrapped_object)
            environment_objects.append(wrapped_object)

        for node_def, node_file, side_length in EnvDefs.boxes:
            wrapped_object = Cube(node_def, node_file, side_length=side_length)
            self._place_object_in_random_position(environment_objects, wrapped_object)
            environment_objects.append(wrapped_object)

        return environment_objects

    def _place_object_in_random_position(self, placed_objects, wrapped_object):
        """
        Sets the shape passed by parameter to a random valid position
        within the parent's node environment.

        :param wrapped_object: the wrapped object with utility functions that is to be placed
        :param placed_objects: the objects that have already been placed
        :return: the node corresponding to the shape
        """
        self.children_field.importMFNode(-1, wrapped_object.node_file)
        shape = self.supervisor.getFromDef(wrapped_object.node_def)
        wrapped_object.webot_object = shape

        x, z = self._generate_random_valid_position(placed_objects, wrapped_object)

        trans_field = shape.getField('translation')
        initial_position = [x, 0.05, z]
        wrapped_object.initial_position = initial_position
        trans_field.setSFVec3f(initial_position)
        shape.resetPhysics()

        return wrapped_object

    def _generate_random_valid_position(self, placed_objects, wrapped_object):
        valid_position_found = False
        min_distance_from_wall = wrapped_object.get_min_distance_from_wall()
        position_x = None
        position_z = None
        while not valid_position_found:
            position_x, position_z = self._get_random_coords_in_arena(min_distance_from_wall)

            if self._intersects_with_robot(position_x, position_z):
                continue

            valid_position_found = True
            for placed_object in placed_objects:
                if placed_object.is_inside_object(position_x, position_z, wrapped_object.get_min_distance_from_wall()):
                    valid_position_found = False
                    continue

        return position_x, position_z

    @staticmethod
    def _get_random_coords_in_arena(min_distance_from_wall):
        floor_x = 1 / 2
        floor_z = 1 / 2
        position_x = np.random.uniform(-floor_x + min_distance_from_wall, floor_x - min_distance_from_wall)
        position_z = np.random.uniform(-floor_z + min_distance_from_wall, floor_z - min_distance_from_wall)
        return position_x, position_z

    def _intersects_with_robot(self, position_x, position_z):
        position_vec = self.robot_initial_position
        return np.sqrt(((position_vec[0] - position_x) ** 2) + ((position_vec[2] - position_z) ** 2)) < 0.1

    def _insert_robot_in_random_position(self):
        trans_field = self.robot.getField('translation')
        x, z = self._get_random_coords_in_arena(0.045)
        self.robot_initial_position = [x, 0.01, z]
        trans_field.setSFVec3f(self.robot_initial_position)


class Maze:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        # initialization helper variables
        self.children_field = self.supervisor.getRoot().getField('children')  # Get list of all the objects of the scene
        self.robot = None

        self.arena_size = np.array(self.supervisor.getFromDef('arena').getField('floorSize').getSFVec2f())
        self.tile_size = np.array([0.25, 0.25])

        self.walls = self._create_walls()

    def reset(self):
        self._respawn_robot()
        self._create_maze()

    def _respawn_robot(self):
        if self.robot is not None:
            self.robot.remove()

        epuck_def, epuck_file = EnvDefs.epuck
        self.children_field.importMFNode(-2, epuck_file)  # Load robot from file and add to second-to-last position
        self.robot = self.supervisor.getFromDef(epuck_def)

        self._insert_robot_in_initial_position()

    def _create_walls(self):
        wall_def, wall_file = EnvDefs.wall
        walls = []

        for i in range(int((self.arena_size[0] / 0.25 + 1) * (self.arena_size[1] / 0.25 + 1))):
            self.children_field.importMFNode(0, wall_file)
            wb_object = self.supervisor.getFromDef(wall_def)

            self._set_object_position(wb_object, 10 + i*0.1, 0)

            walls.append(wb_object)

        return walls

    def _create_maze(self):
        shape = np.ceil(self.arena_size / self.tile_size).astype(int)
        h_walls = np.ones(shape - np.array([1, 0]), dtype=bool)
        v_walls = np.ones(shape - np.array([0, 1]), dtype=bool)
        visited = np.zeros(shape, dtype=bool)

        directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        start = np.array([0, 0])
        visited[tuple(start)] = True
        stack = [start]
        while len(stack) > 0:
            cur = stack.pop()
            neighbors = directions + cur
            ns_x = neighbors[:, 0]
            ns_y = neighbors[:, 1]
            valid_ns = neighbors[(ns_x >= 0) & (ns_x < shape[0]) & (ns_y >= 0) & (ns_y < shape[1])]
            unvisited_ns = valid_ns[~visited[valid_ns[:, 0], valid_ns[:, 1]]]

            if len(unvisited_ns) > 0:
                stack.append(cur)
                idx = np.random.choice(unvisited_ns.shape[0])
                neighbor = unvisited_ns[idx, :]

                direction = neighbor - cur
                wall = np.minimum(cur, neighbor)
                if direction[0] == 0:
                    v_walls[tuple(wall)] = False
                else:
                    h_walls[tuple(wall)] = False

                t_neigh = tuple(neighbor)
                visited[t_neigh] = True
                stack.append(t_neigh)

        zs, xs = np.nonzero(h_walls)
        wall_id = 0
        for i in range(len(xs)):
            x = 0.125 + xs[i] * 0.25 - self.arena_size[0] / 2
            z = 0.25 + zs[i] * 0.25 - self.arena_size[1] / 2

            self._set_object_rotation(self.walls[wall_id], math.pi/2)
            self._set_object_position(self.walls[wall_id], x, z)
            wall_id += 1

        zs, xs = np.nonzero(v_walls)
        for i in range(len(xs)):
            x = 0.25 + xs[i] * 0.25 - self.arena_size[0] / 2
            z = 0.125 + zs[i] * 0.25 - self.arena_size[1] / 2

            self._set_object_rotation(self.walls[wall_id], 0)
            self._set_object_position(self.walls[wall_id], x, z)
            wall_id += 1

    def get_robot(self):
        return self.robot

    def _set_object_position(self, wb_object, x, z):
        trans_field = wb_object.getField('translation')
        position = [x, 0.05, z]
        trans_field.setSFVec3f(position)

    def _set_object_rotation(self, wb_object, angle):
        rot_field = wb_object.getField('rotation')
        rotation = [0.0, 1.0, 0.0, angle]
        rot_field.setSFRotation(rotation)

    def _insert_robot_in_initial_position(self):
        trans_field = self.robot.getField('translation')
        x, z = [-0.375, -0.375]
        self.robot_initial_position = [x, 0.01, z]
        trans_field.setSFVec3f(self.robot_initial_position)

    def _get_random_coords_in_arena(self, min_distance_from_wall):
        floor_x = self.arena_size[0] / 2
        floor_z = self.arena_size[1] / 2
        position_x = np.random.uniform(-floor_x + min_distance_from_wall, floor_x - min_distance_from_wall)
        position_z = np.random.uniform(-floor_z + min_distance_from_wall, floor_z - min_distance_from_wall)
        return position_x, position_z
