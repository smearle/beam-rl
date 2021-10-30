import qdpy
import torch
from torch import nn
from beam_env import BeamEnv
import cv2
from pdb import set_trace as TT

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv1d:
#       torch.nn.init.uniform(m.weight)
        torch.nn.init.orthogonal_(m.weight)

class NCA(nn.Module):
    def __init__(self, n_chan_in, n_chan_out):
        super(NCA, self).__init__()
        n_chan_hid = 32
        self.l1 = nn.Conv1d(n_chan_in, n_chan_hid, 3, 1, 1, padding_mode='circular')
        self.l2 = nn.Conv1d(n_chan_hid, n_chan_hid, 1, 1, 0)
        self.l3 = nn.Conv1d(n_chan_hid, n_chan_out, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        with torch.no_grad():
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            x = torch.sigmoid(self.l3(x))
        return x


n_beams = 30
env = BeamEnv({
    "n_beams": n_beams,
    "paths": 4,
    "render": True,
})

# The NCA will observe its current knowledge of the population (i.e. who might be, and who is definitely not, infected), a set of channels for determining group start- and end-points (from which actions will be drawn), and some auxiliary memory channels
n_grp = 2  # Currently hard-coded in environment.
n_grp_acts = n_grp * 2  # Actions for selecting start- and end-points
n_aux = 11
n_in_chan = 1 + n_aux + n_grp_acts
n_out_chan = n_in_chan - 1
state = torch.zeros(size=(n_in_chan, n_beams), requires_grad=False)
model = NCA(n_in_chan, n_out_chan)
render = True
if render:
    win_name = "nca_render"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
# Episodes.
for e in range(100):
    state[:] = 0
    obs = torch.Tensor(env.reset())
    state[0] = obs
    if render:
        cv2.imshow(win_name, state.numpy())
        cv2.waitKey(1)
    # Environment steps
    for s in range(15):
        # Passes through the NCA
        for i in range(10):
            state[1:] = model(state.unsqueeze(0))
            if render:
                cv2.imshow(win_name, state.numpy())
                cv2.waitKey(1)
        action = torch.argmax(state[-n_grp_acts:], dim=1)
        obs, rew, done, info = env.step(action)
        state[0] = torch.Tensor(obs)
