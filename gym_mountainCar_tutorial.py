import numpy as np 
import gym
import random 
import matplotlib.pyplot as plt

#####################################################################################
# GETTING STARTED WITH OPEN_GYM
#####################################################################################
# This file was written following the tutorial listed in sources. 
# I highly recomend this tutorial to anyone just getting started. 
# The logic here is identical to what he has already written, though
# some of the variable names have been changed. 

#####################################################################################
# Source: 
#####################################################################################
# https://www.youtube.com/watch?v=Gq1Azv_B4-4

# Learning parameters 
learningRate = 0.1
discount = 0.95
episodes = 10000

# Exploration parameters 
exploreThreshold = 0.95
explorDecayRate  = 0.25

# Render values
renderEveryNum = 2000 
statsEveryNum  = 10

# Initialize the environment
env = gym.make("MountainCar-v0")
env.reset()
envRange = env.observation_space.high - env.observation_space.low

# Initialize statistics record
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Initialize state space and policy
stateSize = [20,20] # typically would change per environment
q_table = np.random.uniform(low=2, high=0, size = [20,20] + [env.action_space.n] )

# Function to get descrite space from continuous location
def getDiscreteState(state): 
    state = (state - env.observation_space.low)/(envRange/stateSize[0])
    return tuple(state.astype(np.int))

# Train the model over the number of episodes
for i in range(episodes):

    # Limit render 
    if i % renderEveryNum == 0: 
        render = True
    else: 
        render= False

    # Initialize state 
    currDescriteState = getDiscreteState(env.reset())
    qState = q_table
    done = False

    # Train over single epoch
    episode_reward = 0
    while not done: 

        # Take greedy or random action depending on epoch-dependent threshold
        if np.random.random() <= exploreThreshold: 
            action = np.random.randint(0,env.action_space.n)
        else: 
            action = np.argmax(q_table[currDescriteState])

        # Get current state, reward, and, termination state, and (info) 
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Render if true
        if render: 
            env.render()

        # Map state to state-space representation 
        newDescriteState = getDiscreteState(new_state)

        # Update policy function / q-table
        if not done: 

            # Get the maximum polivy at the next time step
            maxFutureQ = np.max(q_table[newDescriteState])

            # Get the current SS-A value pair
            currentQ = q_table[currDescriteState + (action, ) ]

            # Update policy
            # (1-learningRate) === How much do you want to remember your current value
            # learningRate === How much do you want to replace your current value
            # reward === Current state value from reward
            # discount*maxFutureQ === Current state value from expected reward
            new_q = currentQ*(1-learningRate) + learningRate*( reward + discount*maxFutureQ )

            # Update q-table (policy table)
            q_table[currDescriteState+(action,)] = new_q

        elif new_state[0] >= env.goal_position: 

            # Set to max reward in this situation
            q_table[currDescriteState + (action, )] = 0

        # Update state
        currDescriteState = newDescriteState

    # Update exploration threshold
    exploreThreshold = exploreThreshold*explorDecayRate

    ep_rewards.append(episode_reward)
    if not i % statsEveryNum:
        average_reward = sum(ep_rewards[-statsEveryNum:])/statsEveryNum
        aggr_ep_rewards['ep'].append(i)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-statsEveryNum:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-statsEveryNum:]))
        #print(f'Episode: {i:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {exploreThreshold:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

# Until it explores all possible situations, it may not find the best policy. 
# Unexplored states might be able to have higher rewards than explored states due to
# update of policy when explored.  




