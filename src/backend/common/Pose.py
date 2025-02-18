import math
import random
from src.backend.common.Distributions import UniformDistribution

class Pose:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        """
        Initialize a Pose object.
        :param x: X-coordinate
        :param y: Y-coordinate
        :param angle: Angle in degrees
        """
        self.x = x
        self.y = y
        self.angle = angle

    def set(self, x, y, angle):
        """
        Update the pose with new x, y, and angle values.
        :param x: New X-coordinate
        :param y: New Y-coordinate
        :param angle: New angle in degrees
        """
        self.x = x
        self.y = y
        self.angle = angle

    def get(self):
        """
        Get the current pose as a tuple.
        :return: (x, y, angle)
        """
        return self.x, self.y, self.angle

    def translate(self, dx, dy):
        """
        Translate the pose by dx and dy.
        :param dx: Change in X-coordinate
        :param dy: Change in Y-coordinate
        """
        self.x += dx
        self.y += dy

    def rotate(self, d_angle):
        """
        Rotate the pose by a given angle.
        :param d_angle: Angle to add in degrees
        """
        self.angle = (self.angle + d_angle) % 360

    def distance_to(self, other_pose):
        """
        Calculate the Euclidean distance to another pose.
        :param other_pose: Another Pose object
        :return: Distance between the two poses
        """
        dx = self.x - other_pose.x
        dy = self.y - other_pose.y
        return math.sqrt(dx**2 + dy**2)

    def to_dict(self):
        """
        Return the pose as a dictionary.
        :return: {'x': x, 'y': y, 'angle': angle}
        """
        return {'x': self.x, 'y': self.y, 'angle': self.angle}

    def __repr__(self):
        """
        String representation of the Pose object.
        :return: A string describing the pose.
        """
        return f"Pose(x={self.x}, y={self.y}, angle={self.angle})"

    def __eq__(self, other):
        """
        Compare two Pose objects for equality.
        :param other: Another Pose object
        :return: True if equal, False otherwise
        """
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y) and math.isclose(self.angle, other.angle)

    def as_polar(self):
        """
        Convert the pose to polar coordinates.
        :return: (r, theta), where r is the radius and theta is the angle in degrees
        """
        r = math.sqrt(self.x**2 + self.y**2)
        theta = math.degrees(math.atan2(self.y, self.x))
        return r, theta

    @classmethod
    def generate_random_pose(cls, x_range, y_range, angle_range):
        """
        Generate a Pose with x, y, and angle from Uniform distributions.
        :param x_range: Tuple (min_x, max_x) for x-coordinate
        :param y_range: Tuple (min_y, max_y) for y-coordinate
        :param angle_range: Tuple (min_angle, max_angle) for angle in degrees
        :return: A new Pose instance with random x, y, angle
        """
        x = UniformDistribution(*x_range).random_samples()
        y = UniformDistribution(*y_range).random_samples()
        angle = UniformDistribution(*angle_range).random_samples()
        return x, y, angle