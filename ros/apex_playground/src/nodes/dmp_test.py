#!/usr/bin/python

import rospy
from poppy_msgs.srv import ReachTarget, ReachTargetRequest
from sensor_msgs.msg import JointState
import os
import numpy as np
from explauto.utils import bounds_min_max

from apex_playground.learning.dmp.mydmp import MyDMP


class ErgoDMP(object):
    def __init__(self):
        self.apex_name = os.environ.get("ROS_HOSTNAME").replace("-ergo.local", "").replace("-", "_");

    def move_to(self, point, duration=0.4):
        service = '/{}/poppy_ergo_jr/reach'.format(self.apex_name)
        rospy.wait_for_service(service)
        reach = rospy.ServiceProxy(service, ReachTarget)
        reach_jointstate = JointState(position=point, name=["m{}".format(i) for i in range(1, 7)])
        reach_request = ReachTargetRequest(target=reach_jointstate,
                                           duration=rospy.Duration(duration))
        reach(reach_request)
        rospy.sleep(duration)


if __name__ == "__main__":
    n_dmps = 6
    n_bfs = 7
    timesteps = 30
    max_params = np.array([300.] * n_bfs * n_dmps + [1.] * n_dmps)
    bounds_motors_max = np.array([180, 10, 20, 10, 30, 30])
    bounds_motors_min = np.array([-180, -20, -20, -15, -20, -20])
    dmp = MyDMP(n_dmps=n_dmps, n_bfs=n_bfs, timesteps=timesteps, max_params=max_params)

    mover = ErgoDMP()
    point = [0, 0, 0, 0, 0, 0]
    mover.move_to(list(point), duration=1)
    m = np.random.randn(dmp.n_dmps * dmp.n_bfs + n_dmps) * max_params
    normalized_traj = dmp.trajectory(m)
    normalized_traj = bounds_min_max(normalized_traj, n_dmps * [-1.], n_dmps * [1.])
    traj = ((normalized_traj - np.array([-1.] * n_dmps)) / 2.) * (bounds_motors_max - bounds_motors_min) + bounds_motors_min
    for m in traj:
        mover.move_to(list(m))
