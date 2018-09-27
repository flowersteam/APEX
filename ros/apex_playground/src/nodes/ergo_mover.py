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
        service = '/{}/poppy_ergo_jr/reach'.format(self.apex_name)
        rospy.wait_for_service(service)
        reach = rospy.ServiceProxy(service, ReachTarget)
        reach_jointstate = JointState(position=point, name=["m{}".format(i) for i in range(1,7)])
        reach_request = ReachTargetRequest(target=reach_jointstate,
                                           duration=rospy.Duration(5))
        reach(reach_request)

if __name__ == "__main__":
    mover = ErgoMover()
    while True:
        point = input("Enter a point to reach (e.g. : 5,5,5,5,5,5): \n")
        if not len(point) == 6:
            print("You must enter 6 motor values like '4,4,4,4,4,4,4'")
            continue
        else:
            mover.move_to(list(point))
