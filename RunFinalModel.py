from tensorflow import keras 
import numpy as np
import Environment as environment 
import OneWay as network

model = keras.models.load_model('./OutputModel/model/')
net = network.info()
env = environment.env(net)
max_steps = 150

env._max_episode_steps = max_steps
state = np.array([env.reset(1)])
done = False
iters = 0
rewards = 0

while not done and iters<max_steps:
    action_probs, critic_value = model(state)
    p=np.squeeze(action_probs)
    action = np.argmax(p)
    state, reward, done, info = env.step(action)
    state = np.array([state])
    rewards += reward 
    iters += 1
    for j in range(np.size(state)):
        print("%.3f\t" % state[0,j], end="")
    print()
    
    
if not done: env.close()