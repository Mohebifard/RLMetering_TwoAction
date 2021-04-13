print("%%%%%% Training Model RLMetering_03 %%%%%%\n\n\n")

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import Environment as environment 
import OneWay as network


# create the environment 
net = network.info()
env = environment.env(net)
gamma = 1.0 #0.99 #discount factor 
solved_criterion = 0.21
max_iter_episode = 200

#np.random.seed(seed)
#tf.random.set_seed(seed)

n_inputs = 52
n_actions = 2
n_hidden = 500

#set the policy and value model 
input_layer = layers.Input(shape=(n_inputs,))

mid_player = layers.Dense(n_hidden, activation="relu")(input_layer)
last_player = layers.Dense(n_actions, activation="softmax")(mid_player)

mid_vlayer = layers.Dense(n_hidden, activation="relu")(input_layer)
last_vlayer = layers.Dense(1)(mid_vlayer)

model = keras.Model(inputs = input_layer, outputs=[last_player, last_vlayer])
model.summary()

#set the optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)

#set the initial lists and variables
eps = np.finfo(np.float32).eps.item() 
#loss_function = keras.losses.Huber()
loss_function = keras.losses.MSE
p_losse = []
action_prob_histroy = []
expected_return_history = []
reward_history = []
running_reward = 0
running_reward_history = []
episode_count = 0

while True: # run each episode until done is True
    state = env.reset()
    episode_reward = 0
    
    with tf.GradientTape() as tape:
        
        #play one entire episode 
        for timestep in range(1, max_iter_episode):
            
            #adjust state shape 
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            
            #predict action probabilities and expected returns 
            action_prob, expected_return = model(state)
            expected_return_history.append(expected_return[0,0])
            
            #take a random action from the probabilities 
            action = np.random.choice(n_actions, p=np.squeeze(action_prob))
            action_prob_histroy.append(tf.math.log(action_prob[0, action]))
            
            #apply action and observe reward 
            state, reward, done, _ = env.step(action)
            reward_history.append(reward)
            episode_reward += reward 
            
            if done:
                break
            
        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        running_reward_history.append(running_reward)
            
        #calculate the observed returns 
        observed_returns = []
        discounted_sum = 0
        for r in reward_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            observed_returns.insert(0, discounted_sum)
        
        #normalize observed returns
        observed_returns = np.array(observed_returns)
        observed_returns = (observed_returns - np.mean(observed_returns)) / (np.std(observed_returns) + eps)
        observed_returns = observed_returns.tolist()
        
        #calculate loss values 
        histroy = zip(action_prob_histroy, expected_return_history, observed_returns)
        pmodel_loss = []
        vmodel_loss = []        
        for log_prob, e_v, o_v in histroy:
            advantage = o_v - e_v
            pmodel_loss.append(-log_prob * advantage)
            vmodel_loss.append(loss_function(tf.expand_dims(e_v, 0), tf.expand_dims(o_v, 0)))
            
        #back propagation 
        model_loss_value = sum(pmodel_loss) + sum(vmodel_loss)        
        model_grads = tape.gradient(model_loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(model_grads, model.trainable_variables))
        
        #clear history of the episode
        expected_return_history.clear()
        action_prob_histroy.clear()
        reward_history.clear()
        
    #print episode summary 
    episode_count += 1
    if episode_count % 10 == 0:
        print("running reward %.2f at episode %d" % (running_reward, episode_count))
    
    if running_reward > solved_criterion:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
    
model.save('./OutputModel/model/')
    
plt.plot(running_reward_history)
plt.title("Rewards")
plt.xlabel("Iteration Number")
plt.ylabel("running reward")
plt.show()

    