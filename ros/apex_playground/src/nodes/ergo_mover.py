#!/usr/bin/python

import rospy
import json
from poppy_msgs.srv import ReachTarget, ReachTargetRequest
from sensor_msgs.msg import JointState
import os
from os.path import join
import numpy as np


class ErgoMover(object):
    def __init__(self):
        self.apex_name = os.environ.get("ROS_HOSTNAME").replace("-ergo.local", "").replace("-", "_");
         
    def move_to(self, point):
        rospy.wait_for_service('/{}/poppy_ergo_jr/poppy_ergo_jr_controllers'.format(self.apex_name))
        reach = rospy.ServiceProxy('/{}/poppy_ergo_jr/poppy_ergo_jr_controllers'.format(self.apex_name), ReachTarget)
        reach_jointstate = JointState(position=point)
        reach_request = ReachTargetRequest(target=reach_jointstate,
                                           duration=ropsy.Duration(5))
        reach(reach_request)

if __name__ == "__main__":
    mover = ErgoMover()
    while True:
        point = input("Enter a point to reach (e.g. : 5,5,5,5,5,5): \n")
        point = point.split(",")
        if not len(point) == 6:
            print("You must enter 6 motor values like '4,4,4,4,4,4,4'")
            continue
        else:
            mover.move_to(point)
