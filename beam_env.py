import gym
import numpy as np
from ray.rllib.env.env_context import EnvContext
import cv2
from pdb import set_trace as TT

class BeamEnv(gym.Env):
    def __init__(self, config: EnvContext):
        '''An environment for learning a wireless communications beam-forming policy via group testing.
            n_beams: how many slices in our pie (a discretized circle)
        '''
        render_gui = config.get("render", False)
        n_beams = config.get("n_beams", 16)
        n_paths = config.get("n_paths", 4)
        self.observation_space = gym.spaces.MultiBinary(n_beams)
        # TODO: improve this action spacce?
        # This is a very naive version of the action space which allows for overlapping groups. But overlapping groups
        # are strictly redundant (right?)
        self.action_space = gym.spaces.MultiDiscrete([n_beams, n_beams-2, n_beams, n_beams-2])
        self.n_beams = n_beams
        self.n_paths = n_paths
        self.render_gui = render_gui
        if render_gui:
            self.frame_delay = 100
            self.path_color = [0, 255, 0]
            self.g1_color = [255, 0, 0]
            self.g2_color = [0, 0, 255]
            self.win_name = "beam_render"
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            self.true_img = np.zeros((1, n_beams, 3))
            self.obs_img = np.ones((1, n_beams, 3))
            self.groups_img = np.zeros((2, n_beams, 3))

    def reset(self):
        self.n_step = 0
        if self.n_paths is None:
            # randomly initialize some paths whose location our algorithm will try to pinpoint
            n_paths = np.random.randint(2, 5)
        else:
            # Use a fixed number of paths
            n_paths = self.n_paths
        path_idxs = np.random.choice(self.n_beams, n_paths, replace=False)
        self.state = np.ones(self.n_beams, dtype=bool)
        self.true_state = np.zeros(self.n_beams, dtype=bool)
        self.true_state[path_idxs] = 1
        if self.render_gui:
            self.true_img.fill(0)
            self.true_img[:, path_idxs, :] = self.path_color
            self.obs_img.fill(1)
            self.render()
#           print('True state:')
#           print(self.true_state.astype(int))
        return self.state

    def render(self):
        img = np.vstack((self.true_img, self.groups_img, self.obs_img))
        cv2.imshow(self.win_name, img)
        cv2.waitKey(self.frame_delay)

    def step(self, action):
        '''
        - action: tuple of indices; first is beginning of first group, second is end of first group; 3r
        '''
        # divide our pie up into two groups for testing
        b_1, b_2 = action[0], (action[0] + 1 + action[1]) % self.n_beams
        b_3, b_4 = action[2], (action[2] + 1 + action[3]) % self.n_beams
        def test_group(a, b, true_state, obs, render_idx=None):
            assert a != b  # this would be redundant
            if a < b:
                if render_idx is not None:
                    self.groups_img[render_idx][a:b] = 1
                infected = np.any(true_state[a:b])
                obs[a:b] = infected
            else:
                if render_idx is not None:
                    self.groups_img[render_idx][a:] = self.groups_img[render_idx][:b] = 1
                infected = np.any(np.hstack((true_state[a:], true_state[:b])))
                obs[a:] = obs[:b] = infected
            return obs
        test_result_0 = np.ones(self.n_beams, dtype=bool)
        test_result_1 = np.ones(test_result_0.shape, dtype=bool)
        test_result_0 = test_group(b_1, b_2, self.true_state, test_result_0, render_idx=0 if self.render_gui else None)
        test_result_1 = test_group(b_3, b_4, self.true_state, test_result_1, render_idx=1 if self.render_gui else None)
        new_state = test_result_0 & test_result_1 & self.state
        rew = np.sum(self.state) - np.sum(new_state)  # reward for finding true negatives
        self.state = new_state
        done = np.all(self.state == self.true_state)
        if self.render_gui:
            self.groups_img[0] *= self.g1_color
            self.groups_img[1] *= self.g2_color
            self.render()
            self.obs_img.fill(0)
            self.obs_img[0, self.state, :] = self.path_color
            self.render()
            self.groups_img.fill(0)
#           print('Observed:')
#           print(self.state.astype(int))
        return self.state, rew - 1, done, {}

class Agent(object):
    def __init__(self, env):
        self.env = env

    def play_episode(self):
        obs = self.reset()
        done = False
        self.n_step = 0
        cum_rew = 0
        while not done:
            obs, rew, done, info = self.act(obs)
            cum_rew += rew
            self.n_step += 1
        return cum_rew, self.n_step

    def reset(self):
        return self.env.reset()

    def act(self, obs):
        raise NotImplementedError

class NeuralAgent(Agent):
    def __init__(self, env, nn):
        super().__init__(env)
        self.nn = nn

    def act(self, obs):
        TT()
        return self.nn.compute_actions()

class SequentialAgent(Agent):
    def __init__(self, env):
        super(SequentialAgent, self).__init__(env)

    def act(self, obs):
        '''Test individual beams sequentially.'''
        return self.env.step((self.n_step*2, 0,
                              self.n_step*2+1, 0))

class MergeAgent(Agent):
    '''Divide and conquer. If either group tests positive, break it down the middle into two new groups and recurse,
    then merge the results.'''
    def __init__(self, env):
        super(MergeAgent, self).__init__(env)

    def reset(self):
        self.n_step = 0
        return super(MergeAgent, self).reset()

    def play_episode(self):
        obs = self.reset()
        cum_rew = self.act(obs)
        return cum_rew, self.n_step

    def act(self, obs):
        assert len(obs.shape) == 1
        pos = np.arange(obs.shape[0])
        obs, cum_rew, done, info = self.split_test(pos, obs)
        return cum_rew

    def split_test(self, pos, obs):
        # we know pos is always an incremental sequence
        mid = pos[len(pos) // 2]
        # weird because actions are each group's start, then the __distance__ to its end (not the index of the endpoint)
        g0 = pos[0]
        g0_delta = mid - pos[0] - 1
        g1 = mid
        g1_delta = pos[-1] - mid
        obs, rew, done, info = self.env.step((g0, g0_delta,
                                              g1, g1_delta))
        self.n_step += 1
        # use global pos to match obs (will slice again below with global g0, g1 coords)
        glob_pos = np.arange(len(obs))
        result_0 = obs[g0:g0+g0_delta+1]
        result_1 = obs[g1:g1+g1_delta+1]
        if len(result_0) > 1 and np.any(result_0):
#           assert np.all(result_0)
            obs_0, rew_0, done, info_0 = self.split_test(glob_pos[g0: g0+g0_delta+1],
                                                           obs[g0: g0+g0_delta+1])
            rew += rew_0
            if done:
                return obs, rew, done, info_0
            obs[g0:g0 + g0_delta + 1] = obs_0
        if len(result_1) > 1 and np.any(result_1):
#           assert np.all(result_1)
            obs_1, rew_1, done, info_1 = self.split_test(glob_pos[g1: g1 + g1_delta+1],
                                                           obs[g1: g1 + g1_delta+1])
            rew += rew_1
            if done:
                return obs, rew, done, info_1
            obs[g1:g1 + g1_delta+1] = obs_1

        return obs[pos], rew, done, info


class RandAgent(Agent):
    def __init__(self, env):
       super(RandAgent, self).__init__(env)

    def act(self, obs):
        return self.env.step(self.env.action_space.sample())


if __name__ == '__main__':
    env = BeamEnv(EnvContext(worker_index=0, env_config={
                    'render': True,
                }))
    agents_cls = [MergeAgent, SequentialAgent, RandAgent]
    n_trials = 100
    agents_rew = np.zeros(shape=(len(agents_cls), n_trials))
    agents_steps = np.zeros(shape=agents_rew.shape)
    for agent_i, agent_cls in enumerate(agents_cls):
        agent = agent_cls(env)
        for trial_i in range(n_trials):
            rew, n_steps = agent.play_episode()
            print('done in {} steps, reward {}'.format(n_steps, rew))
            agents_rew[agent_i, trial_i] = rew
            agents_steps[agent_i, trial_i] = n_steps



