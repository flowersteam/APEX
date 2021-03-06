from explauto.utils import bounds_min_max
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rospkg import RosPack
from os.path import join
import json
import numpy as np
import rospy


class GRBFTrajectory(object):
    def __init__(self, n_dims, sigma, steps_per_basis, max_basis):
        self.n_dims = n_dims
        self.sigma = sigma
        self.alpha = - 1. / (2. * self.sigma ** 2.)
        self.steps_per_basis = steps_per_basis
        self.max_basis = max_basis
        self.precomputed_gaussian = np.zeros(2 * self.max_basis * self.steps_per_basis)
        for i in range(2 * self.max_basis * self.steps_per_basis):
            self.precomputed_gaussian[i] = self.gaussian(self.max_basis * self.steps_per_basis, i)
        
    def gaussian(self, center, t):
        return np.exp(self.alpha * (center - t) ** 2.)
    
    def trajectory(self, weights):
        n_basis = len(weights)//self.n_dims
        weights = np.reshape(weights, (n_basis, self.n_dims)).T
        steps = self.steps_per_basis * n_basis
        traj = np.zeros((steps, self.n_dims))
        for step in range(steps):
            g = self.precomputed_gaussian[self.max_basis * self.steps_per_basis + self.steps_per_basis - 1 - step::self.steps_per_basis][:n_basis]
            traj[step] = np.dot(weights, g)
        return np.clip(traj, -1., 1.)


class EnvironmentTranslator(object):
    """
    This class gives sense to all the numerical parameters used by the learning and handles the transformation:
    Huge list of floats <=> meaningful class instances

    Therefore it also stores the joint names/order
    """
    def __init__(self):
        self.rospack = RosPack()
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'bounds.json')) as f:
            self.bounds = json.load(f)
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'general.json')) as f:
            self.params = json.load(f)
        self.bounds_motors_min = np.array([float(bound[0]) for bound in self.bounds['motors']['positions']])
        self.bounds_motors_max = np.array([float(bound[1]) for bound in self.bounds['motors']['positions']])
        self.bounds_sensory_min = [d for space in ['hand', 'joystick_1', 'joystick_2', 'ergo', 'ball', 'light', 'sound', "hand_right", "base", "arena", "obj1", "obj2", "obj3", "rdm1", "rdm2"] for d in [float(bound[0])for bound in self.bounds['sensory'][space]]*10]
        self.bounds_sensory_min = np.array([float(self.bounds['sensory']['ergo'][0][0]), float(self.bounds['sensory']['ball'][0][0])] + self.bounds_sensory_min)
        self.bounds_sensory_max = [d for space in ['hand', 'joystick_1', 'joystick_2', 'ergo', 'ball', 'light', 'sound', "hand_right", "base", "arena", "obj1", "obj2", "obj3", "rdm1", "rdm2"] for d in [float(bound[1])for bound in self.bounds['sensory'][space]]*10]
        self.bounds_sensory_max = np.array([float(self.bounds['sensory']['ergo'][0][1]), float(self.bounds['sensory']['ball'][0][1])] + self.bounds_sensory_max)
        self.bounds_sensory_diff = self.bounds_sensory_max - self.bounds_sensory_min

        # RBF PARAMETERS
        self.motor_dims = 4
        self.steps_per_basis = 6
        self.sigma = self.steps_per_basis // 2
        self.max_basis = 5

        self.trajectory_generator = GRBFTrajectory(self.motor_dims,
                                                   self.sigma,
                                                   self.steps_per_basis,
                                                   self.max_basis)
        # DMP PARAMETERS
        #self.n_dmps = 4
        #self.n_bfs = 7
        self.timesteps = 30
        #self.max_params = np.array([300.] * self.n_bfs * self.n_dmps + [1.] * self.n_dmps)
        #self.motor_dmp = MyDMP(n_dmps=self.n_dmps, n_bfs=self.n_bfs, timesteps=self.timesteps, max_params=self.max_params)
        self.context = {}
        self.config = dict(m_mins=[-1.]*20,
                           m_maxs=[1.]*20,
                           s_mins=[-1.]*312,
                           s_maxs=[1.]*312)

    def trajectory_to_w(self, m_traj):
        raise NotImplementedError
        assert m_traj.shape == (self.timesteps, self.n_dmps)
        normalized_traj = ((m_traj - self.bounds_motors_min) / (self.bounds_motors_max - self.bounds_motors_min)) * 2 + np.array([-1.]*self.n_dmps)
        return self.motor_dmp.imitate(normalized_traj) / self.max_params

    def w_to_trajectory(self, w):
        normalized_traj = self.trajectory_generator.trajectory(np.array(w))
        return ((normalized_traj - np.array([-1.]*self.motor_dims))/2.) * (self.bounds_motors_max - self.bounds_motors_min) + self.bounds_motors_min

    def get_context(self, state):
        return [state.ergo.angle, state.ball.angle]

    def sensory_trajectory_msg_to_list(self, state):
        def flatten(list2d):
            return [element2 for element1 in list2d for element2 in element1]

        self.context = {'ball': state.points[0].ball.angle,
                        'ergo': state.points[0].ergo.angle}
        rospy.loginfo("Context {}".format(self.context))
        
        state_dict = {}
        state_dict['hand'] = flatten([(point.hand.pose.position.x, point.hand.pose.position.y, point.hand.pose.position.z) for point in state.points])
        state_dict['joystick_1'] = flatten([point.joystick_1.axes for point in state.points])
        state_dict['joystick_2'] = flatten([point.joystick_2.axes for point in state.points])
        state_dict['ergo'] = flatten([(point.ergo.angle, float(point.ergo.extended)) for point in state.points])
        state_dict['ball'] = flatten([(point.ball.angle, float(point.ball.extended)) for point in state.points])
        #state_dict['ergo'] = flatten([((point.ergo.angle - self.context['ergo']) / 2., float(point.ergo.extended)) for point in state.points])
        #state_dict['ball'] = flatten([((point.ball.angle - self.context['ball']) / 2., float(point.ball.extended)) for point in state.points])
        state_dict['light'] = [point.color.data for point in state.points]
        state_dict['sound'] = [point.sound.data for point in state.points]

        state_dict['hand_right'] = flatten([(0., 0., 0.) for _ in range(10)])
        state_dict['base'] = flatten([(0., 0., 0.) for _ in range(10)])
        state_dict['arena'] = flatten([(0., 0.) for _ in range(10)])
        state_dict['obj1'] = flatten([(0., 0.) for _ in range(10)])
        state_dict['obj2'] = flatten([(0., 0.) for _ in range(10)])
        state_dict['obj3'] = flatten([(0., 0.) for _ in range(10)])
        

        rdm_power = 0.1
        rdm2_x = list(np.cumsum(rdm_power*(np.random.random(10)-0.5)))
        rdm2_y = list(np.cumsum(rdm_power*(np.random.random(10)-0.5)))
        state_dict['rdm1'] = flatten([(rdm2_x[i], rdm2_y[i]) for i in range(10)])
        
        rdm3_x = list(np.cumsum(rdm_power*(np.random.random(10)-0.5)))
        rdm3_y = list(np.cumsum(rdm_power*(np.random.random(10)-0.5)))
        state_dict['rdm2'] = flatten([(rdm3_x[i], rdm3_y[i]) for i in range(10)])

        assert len(state_dict['hand']) == 30, len(state_dict['hand'])
        assert len(state_dict['joystick_1']) == 20, len(state_dict['joystick_1'])
        assert len(state_dict['joystick_2']) == 20, len(state_dict['joystick_2'])
        assert len(state_dict['ergo']) == 20, len(state_dict['ergo'])
        assert len(state_dict['ball']) == 20, len(state_dict['ball'])
        assert len(state_dict['light']) == 10, len(state_dict['light'])
        assert len(state_dict['sound']) == 10, len(state_dict['sound'])

        # Concatenate all these values in a huge 132-float list
        s_bounded = np.array([self.context['ergo'], self.context['ball']] + [value for space in ['hand', 'joystick_1', 'joystick_2', 'ergo', 'ball', 'light', 'sound', 'hand_right', 'base', 'arena', 'obj1', 'obj2', 'obj3', 'rdm1', 'rdm2'] for value in state_dict[space]])
        s_normalized = ((s_bounded - self.bounds_sensory_min) / self.bounds_sensory_diff) * 2 + np.array([-1.]*312)
        s_normalized = bounds_min_max(s_normalized, 312 * [-1.], 312 * [1.])
        # print "context", s_bounded[:2], s_normalized[:2]
        # print "hand", s_bounded[2:32], s_normalized[2:32]
        # print "joystick_1", s_bounded[32:52], s_normalized[32:52]
        # print "joystick_2", s_bounded[52:72], s_normalized[52:72]
        # print "ergo", s_bounded[72:92], s_normalized[72:92]
        # print "ball", s_bounded[92:112], s_normalized[92:112]
        # print "light", s_bounded[112:122], s_normalized[112:122]
        # print "sound", s_bounded[122:132], s_normalized[122:132]

        return list(s_normalized)

    def matrix_to_trajectory_msg(self, matrix_traj):
        assert matrix_traj.shape == (self.timesteps, self.motor_dims)
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ['l_shoulder_y', 'l_shoulder_x', 'l_arm_z', 'l_elbow_y']
        for point in range(len(matrix_traj)):
            traj.points.append(JointTrajectoryPoint(positions=list(matrix_traj[point]),
                                                    time_from_start=rospy.Duration(float(point)/self.params['recording_rate'])))
        return traj

    def trajectory_msg_to_matrix(self, trajectory):
        matrix = np.array([point.positions for point in trajectory.points])
        assert matrix.shape == (self.timesteps, self.motor_dims)
        return matrix
