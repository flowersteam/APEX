import numpy as np
import gizeh

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.context_environment import ContextEnvironment


class RbfController(object):
    """This controller generates time-bounded action sequences using radial basis functions.
    """

    def __init__(self, *args, n_timesteps, n_action_dims, n_rbf, sdev, **kwargs):

        try:
            import scipy.ndimage
            globals()['scipy.ndimage'] = scipy.ndimage
        except:
            raise ImportError("You need scipy.ndimage to use class {}".format(self.__class__.__name__))

        # The array containing the atoms is created by filtering a multidimensional array
        # containing indicators at centers of atoms.
        # We make it larger to convolve outside of support and we cut it after
        self._bfs_params = np.zeros([int(n_timesteps * 1.25), n_action_dims, n_rbf])
        width = n_timesteps // (n_rbf)
        centers = np.cumsum([width] * n_rbf) + int(width // 4)
        base = np.array(range(n_rbf))
        self._bfs_params[centers, :, base] = 1.
        self._bfs_params = scipy.ndimage.gaussian_filter1d(self._bfs_params,
                                                           sdev,
                                                           mode='constant',
                                                           axis=0)
        self._bfs_params /= self._bfs_params.max()

        self._bfs_params = self._bfs_params[:n_timesteps, :, :]

        self.action_sequence = None

    def act(self, parameters):

        self.action_sequence = np.einsum('ijk,jk->ij', self._bfs_params, parameters)


class FixedEpisodeDynamizer():
    """This actor allows to dynamize an environment for a fixed number of iterations.
    """

    def __init__(self, *args, static_env, n_iter, **kwargs):

        self._static_env = static_env(*args, **kwargs)
        self._n_iter = n_iter

        self.observation_sequence = None
        self.reward_sequence = None

    def reset(self):
        self._static_env.reset()
        self.observation_sequence = np.repeat(self._static_env.observation.reshape(1, -1),
                                               repeats=self._n_iter,
                                               axis=0)
        self.reward_sequence = np.array([self._static_env.reward] * self._n_iter)

    def act(self, action_sequence):
        for i, action in enumerate(action_sequence):
            self._static_env.act(action=action)

            self.observation_sequence[i] = self._static_env.observation

            self.reward_sequence[i] = self._static_env.reward


class ArmBall(object):
    """The Armball environment.
    """

    def __init__(self, *args, object_initial_pose=np.array([0.6, 0.6]),
                 object_rewarding_pose=np.array([-0.6, -0.6]),
                 arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]), object_size=0.1, **kwargs):

        assert arm_lengths.size < 8, "The number of joints must be inferior to 8"
        assert arm_lengths.sum() == 1., "The arm length must sum to 1."

        # We set the parameters
        self._n_joints = arm_lengths.size
        self._arm_lengths = arm_lengths
        self._object_initial_pose = object_initial_pose
        self._object_rewarding_pose = object_rewarding_pose
        self._object_size = object_size
        self._actual_object_pose = self._object_initial_pose
        self._actual_arm_pose = np.zeros(self._arm_lengths.shape)
        self._hand_pos = np.zeros(2)
        self._object_handled = False

        # We set the spaces
        self.observation_space = np.array([[-1, 1]] * (len(self._arm_lengths) + 2))
        self.action_space = np.array([[-1, 1]] * self._n_joints)

        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.steps = None

    def reset(self):

        # We reset the simulation
        self._actual_object_pose = self._object_initial_pose
        self._actual_arm_pose = np.zeros(self._arm_lengths.shape)
        self._object_handled = False
        self.observation = np.concatenate([self._actual_arm_pose, self._actual_object_pose])
        self.steps = 0

        # We compute the initial reward.
        self.reward = np.linalg.norm(self._actual_object_pose - self._object_rewarding_pose, ord=2)

    def act(self, action=np.array([0., 0., 0., 0., 0., 0., 0.])):
        """Perform an agent action in the Environment
        """

        assert action.shape == self.action_space.shape[0:1]
        assert (action >= self.action_space[:, 0]).all()
        assert (action <= self.action_space[:, 1]).all()

        # We compute the position of the end effector
        self._actual_arm_pose = action
        angles = np.cumsum(self._actual_arm_pose)
        angles_rads = np.pi * angles
        self._hand_pos = np.array([np.sum(np.cos(angles_rads) * self._arm_lengths),
                                   np.sum(np.sin(angles_rads) * self._arm_lengths)])

        # We check if the object is handled and we move it.
        if np.linalg.norm(self._hand_pos - self._actual_object_pose, ord=2) < self._object_size:
            self._object_handled = True
        if self._object_handled:
            self._actual_object_pose = self._hand_pos

        # We update observation and reward
        self.observation = np.concatenate([self._actual_arm_pose, self._actual_object_pose])
        self.reward = np.linalg.norm(self._actual_object_pose - self._object_rewarding_pose, ord=2)
        self.steps += 1


class ArmBallRenderer(object):
    """This allows to render the ArmBall Environment
    """

    def __init__(self, *args, width=600, height=400, rgb=True, render_arm=True,
                 arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]), object_size=0.1, env_noise=0., **kwargs):

        self._width = width
        self._height = height
        self._rgb = rgb
        self._env_noise = env_noise

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._render_arm = render_arm

        self._observation_space = np.array([[-1, 1]] * (len(self._arm_lengths) + 2))

        self.rendering = None

    def reset(self):

        if self._rgb:
            self.rendering = np.zeros([self._height, self._width, 3])
            self.rendering[0] = 1
        else:
            self.rendering = np.zeros([self._height, self._width])
            self.rendering[0] = 1

        self.act(observation=np.concatenate([np.zeros(len(self._arm_lengths)), [0, 0]]))
        self._typical_img = self.rendering

    def act(self, observation=np.array([0., 0., 0., 0., 0., 0., 0., .6, .6]),
            render_goal=False, goal=None, render_hand=False):

        assert len(observation) == len(self._arm_lengths) + 2

        # We retrieve arm and object pose
        arm_pose = observation[:-2]
        object_pose = observation[-2:]

        # World parameters
        world_size = 2.
        arm_angles = np.cumsum(arm_pose)
        arm_angles = np.pi * arm_angles
        arm_points = np.array([np.cumsum(np.cos(arm_angles) * self._arm_lengths),
                               np.cumsum(np.sin(arm_angles) * self._arm_lengths)])
        hand_pos = np.array([np.sum(np.cos(arm_angles) * self._arm_lengths),
                             np.sum(np.sin(arm_angles) * self._arm_lengths)])

        # Screen parameters
        screen_width = self._width
        screen_height = self._height
        screen_center_w = np.ceil(self._width / 2)
        screen_center_h = np.ceil(self._height / 2)

        # Ratios
        world2screen = min(screen_width / world_size, screen_height / world_size)

        # Instantiating surface
        surface = gizeh.Surface(width=screen_width, height=screen_height)

        # Drawing Background
        background = gizeh.rectangle(lx=screen_width, ly=screen_height,
                                     xy=(screen_center_w, screen_center_h), fill=(1, 1, 1))
        background.draw(surface)

        # Drawing object
        objt = gizeh.circle(r=self._object_size * world2screen,
                            xy=(screen_center_w + object_pose[0] * world2screen,
                                screen_center_h + object_pose[1] * world2screen),
                            fill=(0, 1, 1))
        objt.draw(surface)

        # Drawing goal
        if render_goal == True:
            objt = gizeh.circle(r=self._object_size * world2screen / 4,
                                xy=(screen_center_w + goal[0] * world2screen,
                                    screen_center_h + goal[1] * world2screen),
                                fill=(1, 0, 0))
            objt.draw(surface)

        # Drawing hand
        if render_hand == True:
            objt = gizeh.circle(r=self._object_size * world2screen / 2,
                                xy=(screen_center_w + hand_pos[0] * world2screen,
                                    screen_center_h + hand_pos[1] * world2screen),
                                fill=(1, 0, 0))
            objt.draw(surface)

        # Drawing arm
        if self._render_arm:
            screen_arm_points = arm_points * world2screen
            screen_arm_points = np.concatenate([[[0., 0.]], screen_arm_points.T], axis=0) + \
                                np.array([screen_center_w, screen_center_h])
            arm = gizeh.polyline(screen_arm_points, stroke=(0, 1, 0), stroke_width=3.)
            arm.draw(surface)

        if self._rgb:
            self.rendering = surface.get_npimage().astype(np.float32)
            self.rendering -= self.rendering.min()
            self.rendering /= self.rendering.max()
            if self._env_noise > 0:
                self.rendering = np.random.normal(self.rendering, self._env_noise)
                self.rendering -= self.rendering.min()
                self.rendering /= self.rendering.max()
        else:
            self.rendering = surface.get_npimage().astype(np.float32).sum(axis=-1)
            self.rendering -= self.rendering.min()
            self.rendering /= self.rendering.max()
            if self._env_noise > 0:
                self.rendering = np.random.normal(self.rendering, self._env_noise)
                self.rendering = np.clip(self.rendering, 0, 1)
                # self.rendering -= self.rendering.min()
                # self.rendering /= self.rendering.max()
            # Added by Adrien, makes training easier
            self.rendering = -self.rendering + 1


class MyArmBall(object):
    """
    This is an example of a static environment that could be implemented
    """

    def __init__(self, *args, arm_lengths=np.array([0.5, 0.3, 0.2]), object_size=0.1, n_rbf=5, sdev=5., n_timesteps=50,
                 render=False, **kwargs):

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._n_rbf = n_rbf
        self._n_timesteps = n_timesteps
        self._sdev = sdev

        # We set the spaces
        self.observation_space = np.array([[-1, 1]] * 2)
        self.action_space = np.array([[-1, 1]] * arm_lengths.shape[0] * n_rbf)

        self._dynamic_environment = FixedEpisodeDynamizer(static_env=ArmBall, n_iter=n_timesteps,
                                                          arm_lengths=arm_lengths, object_size=object_size)
        self._controller = RbfController(n_action_dims=len(arm_lengths), n_rbf=n_rbf,
                                         n_timesteps=n_timesteps, sdev=sdev)
        if render:
            self._renderer = MatplotlibInteractiveRendering(ArmBallRenderer, width=500, height=500,
                                                            rgb=False, object_size=object_size, arm_lengths=arm_lengths)
            self._renderer.reset()

        self.observation = None
        self.hidden_state = None

    def reset(self):

        self._dynamic_environment.reset()

        obs = self._dynamic_environment.observation_sequence

        self.observation = obs[-1, :]
        self.hidden_state = self.observation

    def act(self, action, render=True, **kwargs):

        parameterization = action.reshape(self._arm_lengths.shape[0], self._n_rbf)
        self._controller.act(parameterization)
        action_sequence = np.clip(self._controller.action_sequence, a_min=-1, a_max=1)
        self._dynamic_environment.act(action_sequence)
        self.observation = self._dynamic_environment.observation_sequence[-1, :]
        self.hidden_state = self.observation
        if render:
            for i in range(self._n_timesteps):
                self._renderer.act(observation=self._dynamic_environment.observation_sequence[i], **kwargs)


class MyArmBallObserved(object):
    """
    This is an example of a static environment that could be implemented.
    This one represents an observed ArmBall
    """

    def __init__(self, arm_lengths=np.array([0.5, 0.3, 0.2]), object_size=0.1, n_rbf=5, sdev=5.,
                 n_timesteps=50, env_noise=0, render=False, rgb=True, render_arm=False, **kwargs):

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._n_rbf = n_rbf
        self._n_timesteps = n_timesteps
        self._sdev = sdev

        self._action_space = np.concatenate([-np.ones((arm_lengths.shape[0] * n_rbf, 1)),
                                             np.ones((arm_lengths.shape[0] * n_rbf, 1))], axis=-1)

        self._dynamic_environment = FixedEpisodeDynamizer(static_env=ArmBall, n_iter=n_timesteps,
                                                          arm_lengths=arm_lengths, object_size=object_size, **kwargs)
        self._controller = RbfController(n_action_dims=len(arm_lengths), n_rbf=n_rbf,
                                         n_timesteps=n_timesteps, sdev=sdev)
        if render:
            self._renderer = MatplotlibInteractiveRendering(ArmBallRenderer, width=500, height=500, rgb=rgb,
                                                            object_size=object_size, env_noise=env_noise, arm_lengths=arm_lengths)
            self._renderer.reset()

        self._observer = ArmBallRenderer(rgb=rgb, render_arm=render_arm, object_size=object_size,
                                         env_noise=env_noise, arm_lengths=arm_lengths, **kwargs)
        self._observer.reset()

        self.observation = None
        self.explored_states = None
        self.hidden_state = None

    def reset(self):

        self._dynamic_environment.reset()
        self._observer.reset()

        env_state = self._dynamic_environment.observation_sequence
        self._observer.act(observation=env_state[-1])

        self.observation = self._observer.rendering
        self.hidden_state = env_state[-1, -2:]

    def act(self, action, render=True, **kwargs):

        parameterization = action.reshape(self._arm_lengths.shape[0], self._n_rbf)

        self._controller.act(parameterization)
        action_sequence = np.clip(self._controller.action_sequence, a_min=-1, a_max=1)
        self._dynamic_environment.act(action_sequence)
        env_state = self._dynamic_environment.observation_sequence[-1, :]
        self._observer.act(observation=env_state)
        self.observation = self._observer.rendering
        self.hidden_state = env_state[-2:]

        if render:
            for i in range(self._n_timesteps):
                self._renderer.act(observation=self._dynamic_environment.observation_sequence[i], **kwargs)



class TestNCArmBallEnv(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        # SPACES
        # self.hand = 30 * [0.]

        # Env
        self.env = MyArmBall()
        self.env.reset()

        # COMPUTE PERCEPTION
        self.arm = list(self.env.observation[:3])
        self.ball = list(self.env.observation[-2:])

        # CONTEXT
        self.current_context = list(self.arm + self.ball)

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # SAMPLE DMP
        self.env.reset()
        self.env.act(m, render=False)

        # COMPUTE PERCEPTION
        self.arm = list(self.env.observation[:3])
        self.ball = list(self.env.observation[-2:])

        self.current_context = list(self.arm + self.ball)

        #         print
        #         print "hand", self.hand
        #         print "joystick", self.joystick
        #         print "ergo", self.ergo
        #         print "ball", self.ball
        #         print "light", self.light
        #         print "sound", self.sound
        #         print "current_context", self.current_context

        return self.current_context


class TestArmBallEnv(ContextEnvironment):
    def __init__(self):
        env_cls = TestNCArmBallEnv
        env_conf = dict(m_mins=[-1.] * 3 * 5,
                        m_maxs=[1.] * 3 * 5,
                        s_mins=[-1.] * 5,
                        s_maxs=[1.] * 5)

        context_mode = dict(mode='mcs',
                            context_n_dims=5,
                            context_sensory_bounds=[[-1., -1., -1., -1., -1.], [1., 1., 1., 1., 1.]])

        ContextEnvironment.__init__(self, env_cls, env_conf, context_mode)

    def torsodemo2m(self, m_traj): return self.env.torsodemo2m(m_traj)


class TestNCArmBallObsEnv(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        # SPACES
        # self.hand = 30 * [0.]

        # Env
        self.env = MyArmBallObserved(arm_lengths=np.array([0.5, 0.3, 0.2]), object_size=0.1, n_rbf=5,
                                     width=64, height=64)
        self.env.reset()

        # COMPUTE PERCEPTION
        self.image = self.env.observation

        # CONTEXT
        self.current_context = self.env.observation

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # SAMPLE DMP
        self.env.reset()
        self.env.act(m, render=False)

        # COMPUTE PERCEPTION
        self.image = self.env.observation

        self.current_context = self.env.observation

        return self.current_context


class TestArmBallObsEnv(ContextEnvironment):
    def __init__(self):
        env_cls = TestNCArmBallObsEnv
        env_conf = dict(m_mins=[-1.] * 3 * 5,
                        m_maxs=[1.] * 3 * 5,
                        s_mins=[-1.] * 10,
                        s_maxs=[1.] * 10)

        context_mode = dict(mode='mcs',
                            context_n_dims=10,
                            context_sensory_bounds=[[-1.] * 10, [1.] * 10])

        ContextEnvironment.__init__(self, env_cls, env_conf, context_mode)

    def torsodemo2m(self, m_traj): return self.env.torsodemo2m(m_traj)


if __name__ == '__main__':
    a = MyArmBallObserved()
    a.reset()
    random_action = np.random.uniform(-1, 1, a._action_space.shape[0])
    a.act(random_action, render=False)
