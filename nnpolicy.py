import numpy as np

import torch, time, os, glob
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 2, stride=2, padding=1)
        self.flat_dim = flat_dim = 16 * 3 * 3
        self.critic_linear, self.actor_linear = nn.Linear(flat_dim, 1), nn.Linear(flat_dim, num_actions)

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        hx = x.view(-1, self.flat_dim)
        value, probs = self.critic_linear(hx), F.softmax(self.actor_linear(hx), dim=1)
        return value, probs
    
    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; rew = None
        if len(paths) > 0:
            ckpts = [float(s.split('_')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; rew = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if rew is None else print("\tloaded model: {}".format(paths[ix]))
        return rew