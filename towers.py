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
        reward = self.get_reward(action, verbose)
        state = None ; done = True ; info = {}
        return state, reward, done, info
    
    def get_reward(self, action, verbose=False):
        xi = 0 if action in [1,2] else self.qsize
        yi = 0 if action in [1,0] else self.qsize
        obs_quadrant = self.obs.copy()[:, yi:yi+self.qsize, xi:xi+self.qsize]
        health, agent, small, large, friends, enemies = obs_quadrant # split channels
        health += (0.5-1)*(small*health) # small towers get a health in range (0,0.5)
        health += (1.5-1)*(large*health) # large towers get a health in range (0,1.5)
        agent_health = (self.obs[0]*self.obs[1]).sum()
        
        tower_health = (1-agent)*health
        tower_health += -2*friends*tower_health # friends contribute negative points
        tower_health = tower_health.sum() # sum up
        if verbose: print(xi, yi, 'tower health', tower_health, '\tagent health', agent_health)
        return tower_health if tower_health < agent_health else -1.
    
    def get_observation(self):
        channels = [self.get_health_mask(self.qsize)]
        channels += [self.get_agent_mask(self.qsize)]
        channels += self.get_tower_masks(self.qsize, channels[0])
        channels += self.get_team_masks(self.qsize, channels[0])
        channels[0] += np.abs(channels[1]*np.random.rand())     # add the agent to the health channel
        return np.stack(channels)
    
    @staticmethod
    def get_health_mask(qsize):
        channel = np.zeros((qsize*2, qsize*2))
        for i in range(2):
            for j in range(2):
                ix = (np.random.randint(qsize*i, qsize*(i+1)), np.random.randint(qsize*j, qsize*(j+1)))
                channel[ix] = np.random.rand()
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
        friends = np.random.rand(qsize*2, qsize*2) > 0.5
        enemies = 1-friends # this is a redundant layer...but whatever
        return [friends*tower_exists, enemies*tower_exists]