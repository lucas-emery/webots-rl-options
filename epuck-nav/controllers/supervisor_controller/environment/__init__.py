import numpy as np
from utils.env_objects import Cylinder, Cube


class EnvDefs:
    epuck = ('EPUCK', 'webots_objects/E-puck.wbo')
    cylinders = [
        # node DEF,  node file definition, radius
        ('Cylinder1', 'webots_objects/Cylinder1.wbo', 0.05),
        ('Cylinder2', 'webots_objects/Cylinder2.wbo', 0.05)
    ]
    boxes = [
        # node DEF,  node file definition, side length
        ('Box1', 'webots_objects/Box1.wbo', 0.1),
        ('Box2', 'webots_objects/Box2.wbo', 0.1)
    ]


class SimpleArena:

    def __init__(self, supervisor, robot):
        self.supervisor = supervisor
        self.robot = robot
        # initialization helper variables
        self.robot_initial_position = []
        self.children_field = None

        # Respawn robot in starting position and state
        root_node = self.supervisor.getRoot()  # This gets the root of the scene tree
        self.children_field = root_node.getField(
            'children')  # This gets a list of all the children, ie. objects of the scene
        epuck_def, epuck_file = EnvDefs.epuck
        self.children_field.importMFNode(-2, epuck_file)  # Load robot from file and add to second-to-last position
        self.robot = self.supervisor.getFromDef(epuck_def)

        self._insert_robot_in_random_position()
        self.envorinment_objects = self._populate_environment_objects()

    def get_environment_objects(self):
        return self.envorinment_objects

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
