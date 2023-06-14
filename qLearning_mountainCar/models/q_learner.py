# Dependencies
########################################################################

import numpy as np 
import gym
import random 
import matplotlib.pyplot as plt
import math 
import pandas as pd 
import itertools

import qLearning_mountainCar.utils.utils as util

# Sources
########################################################################
# Starting code:   https://www.youtube.com/watch?v=Gq1Azv_B4-4
# Example values:  https://github.com/svpino/lunar-lander
# Deep Q learning: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

# Experiment class
########################################################################
class QLearner: 

    # Experiment variables 
    expName      = [] 
    expType      = [] 
    expID        = []
    maxEps       = [] 
    maxEpsReached= False 
    solvedThresh = []
    solved       = False 

    # Environment variables
    env      = [] 
    envLow   = []
    envHigh  = []
    envRange = []
    numActions  = [] 

    # qLearning parameters 
    qType    = []
    qTable   = []
    divs     = []
    a        = []
    g        = []
    e        = []
    eDecay   = []

    # stateParams
    sNames   = []
    sFlag    = [] 
    sDivs    = []

    # render options
    render    = False
    rendEvery = [] 

    # modifier functions 
    modStates = []

    # statistics options
    results = {'solved': [], 'episodes': [], 'meanReward': [], 'meanActionVar': [], 'meanStateVar': []}
    trainStats = {'ep': [], 'avg': [], 'max': [], 'min': [], 'thresh': [], 'sVar': [], 'aVar': []}
    statsEvery = 1
    stateRecords = [] 

    def __init__(self, exp, env, qParams, sParams, rParams, statParams, modFunction):

        # Exp parameters
        self.expType         = exp.eType
        self.expID           = exp.eId
        self.expName         = exp.name
        self.maxEps          = exp.maxEps
        self.solvedThresh    = exp.solvedThr
        self.consecutiveWins = exp.consecutiveWins #Add different solved conditions

        # stateParams
        self.sNames = np.array([sParams.stateNames[i] for i in range(len(sParams.stateFlag)) if sParams.stateFlag[i] == 1])
        self.sDivs  = np.array([sParams.stateSize[i]  for i in range(len(sParams.stateFlag)) if sParams.stateFlag[i] == 1])

        # Setup environment ranges for q-map
        self.env      = env.env
        self.envLow   = np.array([env.envLow[i]  for i in range(len(sParams.stateFlag)) if sParams.stateFlag[i] == 1])
        self.envHigh  = np.array([env.envHigh[i] for i in range(len(sParams.stateFlag)) if sParams.stateFlag[i] == 1])
        self.envRange = self.envHigh - self.envLow
        self.numActions  = self.env.action_space.n

        # Read q-learning parmeters
        self.a      = qParams.a
        self.g      = qParams.g
        self.e      = qParams.e
        self.eDecay = qParams.eDecay
        self.qType  = qParams.qType

        # Initialize q-table (implemented as a graph/table that grows upon obervation of a new state. *Signifiantly* reduces memory requirements.)
        if self.qType == "map": 
            self.qTable = util.QMap(self.numActions, self.sDivs, self.envLow, self.envHigh)
        else: 
            print(self.qType +" not implemented ") 

        # Set render functions
        self.render    = rParams.render
        self.rendEvery = rParams.rendEvery 

        # set modifier functions
        self.modStates = modFunction

        # statParams
        self.statsEvery = statParams.statsEvery
        
    def run(self): 

        # Print experiment information
        print("########################################################")
        print("Running experiment " + str(self.expID) + " with stats: ")
        print("########################################################")
        print("states: " + str(self.sNames))
        print("sDivs : " + str(self.sDivs))
        print("alpha : " + str(self.a))
        print("gamma : " + str(self.g))
        print("eDecay: " + str(self.eDecay))
        print("")

        # Initialize statistics record
        ep_reward_sums = []
        ep_rewards     = []
        actionRecord   = []
        exploreThresholds = []

        # Train the model over the number of episodes
        episode = 0 
        while not self.solved and not self.maxEpsReached:  
            # Limit render 
            if self.render == True: 
                if episode % self.rendEvery == 0: 
                    render = True
                else: 
                    render= False

            # Initialize state 
            priorContinuousState = self.env.reset()
            priorContinuousState = self.modStates(priorContinuousState, [], self.sNames)

            # Train over single epoch
            done = False
            episode_reward = 0        
            stateRecord    = []
            while not done: 

                # Take greedy or random action depending on epoch-dependent threshold
                # TODO: (later) make p non-uniform
                if np.random.random() <= self.e: 
                    action = np.random.randint(0, self.numActions)
                else: 
                    action = self.qTable.maxQAction(priorContinuousState)

                # Get current state, reward, and, termination state, and (info) 
                newContinuousState, reward, done, _ = self.env.step(action)
                newContinuousState = self.modStates(newContinuousState, priorContinuousState, self.sNames)

                # Record state variables 
                stateRecord.append(newContinuousState.tolist())
                actionRecord.append([i == action for i in range(self.numActions)]) 

                # Update reward if past the goal  
                if newContinuousState[0] >= self.env.goal_position: 
                    reward = 0
                episode_reward += reward

                # Render if true
                if self.render and render: 
                    self.env.render()

                # Get the maximum polivy at the next time step
                maxFutureQ = self.qTable.maxStateQ(newContinuousState)
                currentQ = self.qTable.getQ(priorContinuousState, action)
                new_q = currentQ*(1-self.a) + self.a*( reward + self.g*maxFutureQ )
                self.qTable.setQ(priorContinuousState, action, new_q)

                # Update state
                priorContinuousState = newContinuousState

            # Update exploration threshold
            self.e = self.e*self.eDecay

            # Update win threshold
            ep_rewards.append(reward)
            ep_reward_sums.append(episode_reward)
            exploreThresholds.append(self.e)

            # Evaluate stopping criteria
            numConsecutiveWins = sum( [ i >= self.solvedThresh for i in ep_rewards[-self.consecutiveWins:]])
            if numConsecutiveWins >= self.consecutiveWins: 
                self.solved = True 
            if episode > self.maxEps: 
                self.maxEpsReached = True

            # Print statistics per increment
            # TODO: test
            if ep_rewards[-1:] >= [self.solvedThresh]:
                print("episode: {}; wins: {}; rewards: {}".format(episode, numConsecutiveWins, ep_rewards[-self.consecutiveWins:]))

            if self.solved == True or self.maxEpsReached == True: 
                # Save experiment solution statistics
                self.results['solved'].append(self.solved)
                self.results['episodes'].append(episode)
                self.results['meanReward'].append(np.mean(ep_reward_sums[-self.statsEvery:]))
                self.results['meanActionVar'].append(np.var( np.matrix(stateRecord)[-self.statsEvery:, :], axis=0))
                self.results['meanStateVar'].append( np.var( np.matrix(actionRecord)[-self.statsEvery:, :], axis=0))
            

            # Save episode statistics
            if not episode % self.statsEvery:
                
                self.stateRecords.append(stateRecord)
                
                self.trainStats['ep'].append(episode)
                self.trainStats['avg'].append(sum(ep_reward_sums[-self.statsEvery:])/self.statsEvery)
                self.trainStats['max'].append(max(ep_reward_sums[-self.statsEvery:]))
                self.trainStats['min'].append(min(ep_reward_sums[-self.statsEvery:]))
                self.trainStats['thresh'].append(np.mean(exploreThresholds[-self.statsEvery:]))
                self.trainStats['sVar'].append(np.var(np.matrix(stateRecord)[-self.statsEvery:, :], axis=0))
                self.trainStats['aVar'].append(np.var(np.matrix(actionRecord)[-self.statsEvery:, :], axis=0))

            # Increment episode
            episode += 1 
        
        # End experiment
        self.env.close()
        
    # Save experiment results to a given file
    def saveResults(self, path): 

        # Save training stats plot 
        fig, ax = plt.subplots(1,3)

        ax[0].plot(self.trainStats['ep'], self.trainStats['max'], label="max rewards")
        ax[0].title.set_text("max rewards")

        ax[1].plot(self.trainStats['ep'], self.trainStats['avg'], label="average rewards")
        ax[1].title.set_text("average rewards")    
        
        ax[2].plot(self.trainStats['ep'], self.trainStats['min'], label="min rewards")
        ax[2].title.set_text("min rewards")

        fig.savefig("./{}/Exp_{}_fig.png".format(path, str(self.expID))) 
        np.save("./{}/Exp_{}_stats.npy".format(path, str(self.expID)), self.trainStats) 

        # Save training results 
        np.save("./{}/Exp_{}_results.npy".format(path, str(self.expID)), self.results) 
        np.save("./{}/Exp_{}_q.npy".format(   path, str(self.expID)), self.qTable.saMap) 
        np.save("./{}/Exp_{}_stateRecord.npy".format(   path, str(self.expID)), self.stateRecords) 

    # Function to reset all variables
    def clear(self): 

        self.trainStats['ep'] = []
        self.trainStats['avg'] = []
        self.trainStats['max'] = []
        self.trainStats['min'] = []
        self.trainStats['thresh'] = []
        self.trainStats['sVar'] = []
        self.trainStats['aVar'] = []

        self.results['solved'] = []
        self.results['episodes'] = []
        self.results['meanReward'] = []
        self.results['meanActionVar'] = []
        self.results['meanStateVar'] = []

        # Experiment variables 
        self.expName      = [] 
        self.expType      = [] 
        self.expID        = []
        self.maxEps       = [] 
        self.maxEpsReached= False 
        self.solvedThresh = []
        self.solved       = False 

        # Environment variables
        self.env      = [] 
        self.envLow   = []
        self.envHigh  = []
        self.envRange = []
        self.numActions  = [] 

        # qLearning parameters 
        self.qType    = []
        self.qTable.clear()

        self.qTable   = []
        self.divs     = []
        self.a        = []
        self.g        = []
        self.e        = []
        self.eDecay   = []

        # stateParams
        self.sNames   = []
        self.sFlag    = [] 
        self.sDivs    = []

        # render options
        self.render    = False
        self.rendEvery = [] 

        # modifier functions 
        self.modStates = []

        # statistics options
        self.results = {'solved': [], 'episodes': [], 'meanReward': [], 'meanActionVar': [], 'meanStateVar': []}
        self.trainStats = {'ep': [], 'avg': [], 'max': [], 'min': [], 'thresh': [], 'sVar': [], 'aVar': []}
        self.statsEvery = 1

# Until it explores all possible situations, it may not find the best policy. 
# Unexplored states might be able to have higher rewards than explored states due to
# update of policy when explored.  


