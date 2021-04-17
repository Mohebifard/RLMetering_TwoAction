from tensorflow import keras 
import numpy as np
import Environment as environment 
import OneWay as network
import LPOptimization as LP
import CTM
import matplotlib.pyplot as plt

model = keras.models.load_model('./OutputModel/model/')
net = network.info()
env = environment.env(net)
max_steps = 150

x=[]
env._max_episode_steps = max_steps
state = np.array(env.reset(1))

#simulation 
sim = CTM.simulation()
x_sim, y_sim = sim.xyz(max_steps, net.C, net.Cg, net.Cs, net.CI, net.Q, net.N, net.D, net.W, net.AC, state, True)

#LP model
x_opt, y_opt, z_opt, status_opt = LP.Solve(max_steps, net.C, net.Cg, net.Cs, net.CI, net.Q, net.N, net.D, net.W, net.AC, state, True, True)    
print("Objective function of the optimization model: %.2f\n\n " % (z_opt/(150 * net.Q[net.Cs[0]-1])))

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

    for j in range(np.size(state)):
        print("%.2f\t" % x[-1][j], end="")
    print()

print("\n\n\nTotal rewards (occupancy): %.2f" % (rewards))

c_sim = np.sum(x_sim[:, net.Cs-1], axis=1)
x = np.array(x)
c_RL = np.sum(x[:, net.Cs-1], axis=1)
c_opt = np.sum(x_opt[:, net.Cs-1], axis=1)

c_outflow = np.vstack((np.cumsum(c_sim), np.cumsum(c_RL), np.cumsum(c_opt)))

plt.plot(c_outflow.T)
plt.title("Cumulative Outflow")
plt.xlabel("Time Steps")
plt.ylabel("Outflow")
plt.legend(["Simulation (%.0f)" % (np.sum(c_sim)*450), "RL (%.0f)" % (np.sum(c_RL)*450), "LP (%.0f)" % (np.sum(c_opt)*450)])
plt.show()


