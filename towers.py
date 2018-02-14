import numpy as np
from rtsenv import RTSEnv

class Towers(RTSEnv):
    def __init__(self, quadrant_size=5):
        self.qsize = quadrant_size
        self.labels = ['health', 'agent', 'small', 'large', 'friends', 'enemies']

    def seed(self, seed=0):
        np.random.seed(seed)
        
    def reset(self):
        self.obs = self.get_observation()
        return self.obs.copy()

    def step(self, action, verbose=False):
        qsize = self.qsize ; obs = self.obs
        health, agent, small, large, friends, enemies = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        agent_health = (health*agent).sum()
        xi = 0 if action in [0,3] else qsize
        yi = 0 if action in [0,1] else qsize
        tower_health = ((1-agent[xi:xi+qsize,yi:yi+qsize])*health[xi:xi+qsize,yi:yi+qsize]).sum()
        if verbose: print('tower health', tower_health, '\tagent health', agent_health)
        reward = tower_health if tower_health < agent_health else -3.
        state = None ; done = True ; info = {}
        return state, reward, done, info
    
    def get_observation(self):
        channels = [self.get_health_mask(self.qsize)]
        channels += [self.get_agent_mask(self.qsize)]
        channels += self.get_tower_masks(self.qsize, channels[0])
        channels += self.get_team_masks(self.qsize, channels[0])
        channels[0] += np.abs(channels[1]*np.random.randn())     # add the agent to the health channel
        channels[0] += (0.5-1)*(channels[2]*channels[0]) # small towers get a health in range (0,0.5)
        channels[0] += (1.5-1)*(channels[3]*channels[0]) # large towers get a health in range (0,1.5)
        return np.stack(channels)
    
    @staticmethod
    def get_health_mask(qsize):
        channel = np.zeros((qsize*2, qsize*2))
        for i in range(2):
            for j in range(2):
                ix = (np.random.randint(qsize*i, qsize*(i+1)), np.random.randint(qsize*j, qsize*(j+1)))
                channel[ix] = np.random.randn()
        return channel
    
    @staticmethod
    def get_agent_mask(qsize): # agent will always be in the middle. model should learn to ignore
        channel = np.zeros((qsize*2, qsize*2)) ; channel[qsize,qsize] = 1
        return channel
    
    @staticmethod
    def get_tower_masks(qsize, health_channel):
        tower_exists = health_channel != 0
        small = np.random.rand(qsize*2, qsize*2) > 0.5
        large = 1-small # this is a redundant layer...but whatever
        return [small*tower_exists, large*tower_exists]
    
    @staticmethod
    def get_team_masks(qsize, health_channel):
        tower_exists = health_channel != 0
        friends = health_channel < 0
        enemies = health_channel > 0 # this is a redundant layer...but whatever
        return [friends, enemies] #[team1*tower_exists, team2*tower_exists]