import numpy as np
import CTM as CTM 

class env():
    def __init__(self, net):
        self.observation_space = np.array([len(net.C)])
        self.action_space = 2
        self.sim = CTM.simulation()
        self.net = net
        self.Time = 0
        
    def reset(self, seed = -1):
        
        if seed > 0:
            np.random.seed(seed)
            self.state = np.zeros(len(self.net.C))
        else:
            self.state = np.random.random(len(self.net.C))*0.10
            self.state[self.net.Cg-1] = self.state[self.net.Cg-1] * 0.01 
                
        self.Time = 0
        return self.state 
    
    def step(self, action):
        x, _, = self.sim.xyz(2, self.net.C, self.net.Cg, 
                               self.net.Cs, self.net.CI, self.net.Q, self.net.N, 
                               self.net.D[self.Time:,:], self.net.W[self.Time:,:], 
                               self.net.AC, self.state, True, action)
        
        density = x[2-1,:]/self.net.N
        self.state = np.array(density)
        
        reward = np.sum([density[self.net.Cs-1]])
        
        done = False
        self.Time += 1
        if (self.Time >= self.net.T-1):
            done = True 
        
        info = "System Time of {}".format(self.Time)
        
        return [self.state, reward, done, info]
    
        