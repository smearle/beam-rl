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
        n_beams = config.get("n_beams", 64)
        self.observation_space = gym.spaces.MultiBinary(n_beams)
        self.action_space = gym.spaces.MultiDiscrete([n_beams, n_beams-1])
        self.n_beams = n_beams
        self.render_gui = render_gui
        if render_gui:
            self.frame_delay = 100
            self.path_color = [0, 255, 0]
            self.g1_color = [0, 0, 0]
            self.g2_color = [255, 255, 255]
            self.win_name = "beam_render"
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            self.true_img = np.zeros((1, n_beams, 3))
            self.obs_img = np.ones((1, n_beams, 3))
            self.act_img = np.zeros((1, n_beams, 3))

    def reset(self):
        n_paths = np.random.randint(1, 4)
        # randomly initialize some paths whose location our algorithm will try to pinpoint
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
        img = np.vstack((self.true_img, self.act_img, self.obs_img))
        cv2.imshow(self.win_name, img)
        cv2.waitKey(self.frame_delay)

    def step(self, action):
        # divide our pie up into two big slices and do group testing
        b_1, b_2 = action[0], (action[0] + action[1]) % self.n_beams
        b_1, b_2 = sorted([b_1, b_2])
        test_result = np.zeros(self.n_beams, dtype=bool)
        test_result[b_1:b_2] = np.any(self.true_state[b_1:b_2])
        test_result[:b_1] = test_result[b_2:] = np.any(self.true_state[:b_1]) or np.any(self.true_state[b_2:])
        self.state = test_result & self.state
        done = np.all(self.state == self.true_state)
        if self.render_gui:
            self.act_img.fill(0)
            self.act_img[0, b_1:b_2] = self.g1_color
            self.act_img[0, :b_1] = self.act_img[0, b_2:] = self.g2_color
            self.render()
            self.obs_img.fill(0)
            self.obs_img[0, self.state, :] = self.path_color
            self.render()
#           print('Observed:')
#           print(self.state.astype(int))
        return self.state, -0.1, done, {}

if __name__ == '__main__':
    env = BeamEnv(EnvContext(worker_index=0, env_config={
                    'render': True,
                }))
    for i in range(100):
        n_steps = 0
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            n_steps += 1
        print('done in {} steps'.format(n_steps))
