import json
from rospkg import RosPack
from apex_playground.msg import CircularState
from os.path import join
from numpy import arctan2, sqrt, pi
import csv
import time


class EnvironmentConversions(object):
    def __init__(self):
        self.rospack = RosPack()
        self.last_angle = None
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'environment.json')) as f:
            self.params = json.load(f)
        self.filename = '~/ball_trajectory.csv'

    def save_ball_pos(self, x, y):
		with open(self.filename, 'a') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow([str(time.time()), str(x), str(y)])

    def get_state(self, ball_center, arena_center, ring_radius):
        """
        Reduce the current joint configuration of the ergo in (angle, theta)
        :return: a CircularState
        """
        x, y = (ball_center[0] - arena_center[0], ball_center[1] - arena_center[1])
        self.save_ball_pos(x, y)
        elongation = sqrt(x*x + y*y)
        theta = arctan2(y, x)
        state = CircularState()
        state.extended = elongation > ring_radius
        state.angle = theta
        return state

    def ball_to_color(self, state):
        """
        Reduce the given 2D ball position to color
        :param state: the requested circular state of the ball
        :return: hue value designating the color in [0, 255]
        """
        max_speed = 0.25
        min_speed = 0.07
        if self.last_angle is None:
            hue = 0
        else:
            distance = abs(state.angle-self.last_angle)
            speed = max(min_speed, min(max_speed, min(distance, 2*pi-distance)))
            hue = int((speed - min_speed)/(max_speed - min_speed)*255)
        self.last_angle = state.angle
        return hue

    def ball_to_sound(self, state):
        """
        Reduce the given 2D ball position to sound
        :param state: the requested circular state of the ball
        :return: sound float designating the color within the same range than the circular state angle
        """
        return state.angle if state.extended else 0.
