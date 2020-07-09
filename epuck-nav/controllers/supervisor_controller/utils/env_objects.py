import numpy as np


class EnvObject:
    def __init__(self, node_def, node_file, webot_object=None):
        self.node_def = node_def
        self.node_file = node_file
        self.webot_object = webot_object
        self.initial_position = []

    def is_inside_object(self, point_x, point_z):
        raise NotImplementedError

    def get_min_distance_from_wall(self):
        raise NotImplementedError


class Cylinder(EnvObject):
    def __init__(self, node_def, node_file, radius, webot_object=None):
        super(Cylinder, self).__init__(node_def, node_file, webot_object)
        self.radius = radius

    def is_inside_object(self, point_x, point_z):
        if not self.initial_position:
            raise ValueError("The cylinder object fow which collision is being checked has not been instantiated")
        coordinates = self.initial_position
        return np.sqrt(((coordinates[0] - point_x)**2)+((coordinates[2] - point_z)**2)) < self.radius

    def get_min_distance_from_wall(self):
        return self.radius


class Cube(EnvObject):
    def __init__(self, node_def, node_file, side_length, webot_object=None):
        super(Cube, self).__init__(node_def, node_file, webot_object)
        self.side_length = side_length

    def is_inside_object(self, point_x, point_z):
        if not self.initial_position:
            raise ValueError("The cube object fow which collision is being checked has not been instantiated")

        coordinates = self.initial_position
        if point_x < coordinates[0] - self.side_length:
            return False
        if point_x > coordinates[0] + self.side_length:
            return False
        if point_z < coordinates[2] - self.side_length:
            return False
        if point_z > coordinates[2] + self.side_length:
            return False
        return True

    def get_min_distance_from_wall(self):
        return self.side_length

