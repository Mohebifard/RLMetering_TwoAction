import numpy as np
import CTM as CTM 

class env():
    def __init__(self, net):
        self.observation_space = np.array([len(net.C)])
        self.action_space = 8
        self.sim = CTM.simulation()
        self.net = net
        self.Time = 0
        
    def reset(self, seed = -1):
        
        if seed > 0:
            np.random.seed(seed)
            self.state = np.zeros(len(self.net.C))
        else:
            self.state = np.random.random(len(self.net.C))*0.1
            self.state[self.net.Cg-1] = self.state[self.net.Cg-1] * 0.01 
                
        self.Time = 0
        return self.state 
    
    def step(self, action):
        x, _, z = self.sim.xyz(2, self.net.C, self.net.Cg, 
                               self.net.Cs, self.net.CI, self.net.Q, self.net.N, 
                               self.net.D[self.Time:,:], self.net.W[self.Time:,:], 
                               self.net.AC, self.state, True, action)
        
        self.state = np.array(x[1,:])
        observation = np.array(self.state)
        reward  = np.array(z)
        self.Time += 1
        
        done = False
        if (self.Time >= self.net.T-1):
            done = True 
        
        info = "System Time of {}".format(self.Time)
        
        return [observation, reward, done, info]
    
        
            
            
        
        
        
        
        
        
    
        