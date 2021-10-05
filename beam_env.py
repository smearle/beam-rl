import gym
import numpy as np
from ray.rllib.env.env_context import EnvContext
from pdb import set_trace as TT

class BeamEnv(gym.Env):
    def __init__(self, config: EnvContext):
        '''An environment for learning a wireless communications beam-forming policy via group testing.
            n_beams: how many slices in our pie (a discretized circle)
        '''
        render = config.get("render", False)
        n_beams = config.get("n_beams", 64)
        self.observation_space = gym.spaces.MultiBinary(n_beams)
        self.action_space = gym.spaces.MultiDiscrete([n_beams, n_beams])
        self.n_beams = n_beams
        self.render = render

    def reset(self):
        n_paths = np.random.randint(1, 4)
        # randomly initialize some paths whose location our algorithm will try to pinpoint
        path_idxs = np.random.choice(self.n_beams, n_paths, replace=False)
        self.state = np.ones(self.n_beams, dtype=bool)
        self.true_state = np.zeros(self.n_beams, dtype=bool)
        self.true_state[path_idxs] = 1
        if self.render:
            print('True state:')
            print(self.true_state.astype(int))
        return self.state

    def step(self, action):
        # divide our pie up into two big slices and do group testing
        b_1, b_2 = action
        b_1, b_2 = sorted([b_1, b_2])
        test_result = np.zeros(self.n_beams, dtype=bool)
        test_result[b_1:b_2] = np.any(self.true_state[b_1:b_2])
        test_result[:b_1] = test_result[b_2:] = np.any(self.true_state[:b_1]) or np.any(self.true_state[b_2:])
        self.state = test_result & self.state
        done = np.all(self.state == self.true_state)
        if self.render:
            print('Observed:')
            print(self.state.astype(int))
        return self.state, -0.1, done, {}

if __name__ == '__main__':
    env = BeamEnv(render=False)
    for i in range(100):
        n_steps = 0
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            n_steps += 1
        print('done in {} steps'.format(n_steps))
