from tensorflow import keras 
import numpy as np
import Environment as environment 
import OneWay as network
import LPOptimization as LP

model = keras.models.load_model('./OutputModel/model/')
net = network.info()
env = environment.env(net)
max_steps = 150

x=[]
env._max_episode_steps = max_steps
state = np.array(env.reset(1))

#LP model
x_opt, y_opt, z_opt, status_opt = LP.Solve(max_steps, net.C, net.Cg, net.Cs, net.CI, net.Q, net.N, net.D, net.W, net.AC, state, True, False)    
print("Objective function of the optimization model: %.2f " % (z_opt/(150 * net.Q[net.Cs[0]-1])))

x.append(state*net.N)
done = False
iters = 0
rewards = 0

while not done and iters<max_steps-1:
    action_probs, critic_value = model(np.array([state]))
    p=np.squeeze(action_probs)
    action = np.argmax(p)
    state, reward, done, info = env.step(action)
    state = np.array(state)
    x.append(state*net.N)
    rewards += reward 
    iters += 1

    #for j in range(np.size(state)):
        #print("%.2f\t" % x[-1][j], end="")
    #print()

print("\n\n\nTotal rewards (occupancy): %.2f" % (rewards))


