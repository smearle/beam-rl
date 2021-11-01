import qdpy
import torch as th
from torch import nn
from beam_env import BeamEnv
import cv2
from pdb import set_trace as TT

def init_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv1d:
#       th.nn.init.uniform_(m.weight, 0)
        th.nn.init.orthogonal_(m.weight)


class NCA(nn.Module):
    def __init__(self, n_chan_in, n_chan_out):
        super(NCA, self).__init__()
        n_chan_hid = 32
        self.l1 = nn.Conv1d(n_chan_in, n_chan_hid, 3, 1, 1, padding_mode='circular')
        self.l2 = nn.Conv1d(n_chan_hid, n_chan_hid, 1, 1, 0)
        self.l3 = nn.Conv1d(n_chan_hid, n_chan_out, 1, 1, 0)
        self.layers = [self.l1, self.l2, self.l3]
#       self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = th.relu(self.l1(x))
            x = th.relu(self.l2(x))
            x = th.sigmoid(self.l3(x))
        return x


def get_weights(nn):
    """
    Use to get flat vector of weights from PyTorch model
    """
    init_params = []
    for lyr in nn.layers:
        init_params.append(lyr.weight.view(-1).numpy())
        init_params.append(lyr.bias.view(-1).numpy())
    init_params = np.hstack(init_params)
    print("number of initial NN parameters: {}".format(init_params.shape))

    return init_params

def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False

def set_weights(nn, weights):
    weights = np.array(weights)
#   if ALGO == "ME":
#       # then our nn is contained in the individual
#       individual = weights  # I'm sorry mama
#       return individual.model
    with th.no_grad():
        n_el = 0
        for layer in nn.layers:
            l_weights = weights[n_el : n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = th.nn.Parameter(th.Tensor(l_weights))
            layer.weight.requires_grad = False
            b_weights = weights[n_el : n_el + layer.bias.numel()]
            n_el += layer.bias.numel()
            b_weights = b_weights.reshape(layer.bias.shape)
            layer.bias = th.nn.Parameter(th.Tensor(b_weights))
            layer.bias.requires_grad = False
    return nn


# The NCA will observe its current knowledge of the population (i.e. who might be, and who is definitely not, infected), a set of channels for determining group start- and end-points (from which actions will be drawn), and some auxiliary memory channels
n_grp = 2  # Currently hard-coded in environment.
n_grp_acts = n_grp * 2  # Actions for selecting start- and end-points
n_aux = 11
n_in_chan = 1 + n_aux + n_grp_acts
n_out_chan = n_in_chan - 1
# Episodes.
def simulate_episodes(env, model):
    state = th.zeros(size=(n_in_chan, n_beams), requires_grad=False)
    rew = 0
    n_eps = 10
    for i in range(n_eps):
        rew += simulate_episode(env, model, state)
    rew /= n_eps
    return rew

def simulate_episode(env, model, state):
    th.normal(th.zeros(state.shape).fill_(0.5), th.zeros(state.shape).fill_(0.5), out=state)
    obs = th.Tensor(env.reset())
    state[0] = obs
    net_rew = 0
    if args.render:
        cv2.imshow(win_name, state.numpy())
        cv2.waitKey(env.frame_delay)
    # Environment steps
    for s in range(30):
        # Passes through the NCA
        for i in range(10):
            state[1:] = model(state.unsqueeze(0))
#           state = th.heaviside(state * 2 - 1, th.Tensor([0]))
            if args.render:
                cv2.imshow(win_name, state.numpy())
                cv2.waitKey(env.frame_delay)
        action = th.argmax(state[-n_grp_acts:], dim=1)
        obs, rew, done, info = env.step(action)
        net_rew += rew
        if done:
            break
        state[0] = th.Tensor(obs)
    return -net_rew

def evaluate(sol):
    set_weights(model, sol)
    rew = simulate_episodes(env, model)
    return [rew], [0.5]

from qdpy.algorithms import CMAES, TQDMAlgorithmLogger
from qdpy.containers import Grid
import numpy as np
import argparse
import pickle

args = argparse.ArgumentParser()
args.add_argument('-r', '--render', action='store_true')
args.add_argument('-e', '--evaluate', action='store_true')
args = args.parse_args()
if args.render:
    win_name = "nca_render"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
n_beams = 30
env_config = {
    "n_beams": n_beams,
    "paths": 4,
    "render": args.render,
}

model = NCA(n_in_chan, n_out_chan)
env = BeamEnv(env_config)
set_nograd(model)
if args.evaluate:
    with open("logs/final.p", "rb") as f:
        data = pickle.load(f)
    # ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.
    grid = data['container']
    for sol in grid:
        evaluate(sol)
else:
    init_weights = get_weights(model)
    grid = Grid(shape=(1), max_items_per_bin=100, fitness_domain=[(-np.inf, np.inf),], features_domain=[(0,100),])
    grid.nb_items_per_bin
    algo = CMAES(container=grid, budget=10000, dimension=len(init_weights))
    logger = TQDMAlgorithmLogger(algo, log_base_path="logs")
    result = algo.optimise(evaluate)