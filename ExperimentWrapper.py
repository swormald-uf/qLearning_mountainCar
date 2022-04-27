# Dependencies
########################################################################

import numpy as np 
import gym
import random 
import matplotlib.pyplot as plt
import math 
import pandas as pd 
import itertools
import copy 

import qExperimentClasses as ql 

# Sources
########################################################################
# Starting code:   https://www.youtube.com/watch?v=Gq1Azv_B4-4
# Example values:  https://github.com/svpino/lunar-lander
# Deep Q learning (TODO): https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf 

# Define Classes
########################################################################

# Define classes to enable dot indexing

# container for state definition with bins
class stateSetClass: 
    stateNames= ["x", "vel", "px", "pvel", "dx", "dvel"]    
    stateFlag = [  1,     1,    1,      1,    1,      1]
    stateSize = [ 20,    20,   20,     20,   20,     20]
    def __init__(self, inputStateFlag, divs):
        self.stateFlag = inputStateFlag
        self.stateSize = [ divs, divs, divs, divs, divs, divs]    

#  container for experiment variables
class expClass: 
    eType = []
    eId = []
    name = []
    maxEps = []
    solvedThr = []
    consecutiveWins = []

# container for environment variables
class envClass: 
    # Setup environment ranges for q-map
    env = []
    envLow = []
    envHigh = []

# container for q-learning algorithm parameters 
class qParamsClass: 
    a = []
    g = []
    e = []
    eDecay = []
    qType = []

# container for rendering parameters
class rParamsClass: 
    render = []
    rendEvery = []

# state parameters class
class statParamsClass: 
    statsEvery = []

# Define Functions
########################################################################

# filter function to add output states and feedback modalities from the Mountain Car environment
# Note: This function is specific to the environment. Review the "qLearningClasses.py" file for
# details on usage in case you want to make your own. 
def filterProperties(inputState, priorState, stateNames): 
    inputState = inputState.tolist()
    stateNames = stateNames.tolist()
    if "px" in stateNames: 
        i = stateNames.index("px")
        if priorState != []: 
            inputState.append(priorState[0])
        else:
            inputState.append(inputState[0])
             
    if "pvel" in stateNames: 
        i = stateNames.index("pvel")
        if priorState != []: 
            inputState.append(priorState[1])
        else:
            inputState.append(inputState[1])

    if "dx" in stateNames: 
        i = stateNames.index("dx")
        if priorState != []: 
            inputState.append(inputState[0] - priorState[0])
        else:
            inputState.append(0)

    if "dvel" in stateNames: 
        i = stateNames.index("dvel")
        if priorState != []: 
            inputState.append(inputState[1] - priorState[1])
        else:
            inputState.append(0)

    return np.array(inputState)


# Initialize training and learning parameters
########################################################################

#Details 
path = "mountain_car_exp4"

# Learning parameter options 
statDivs        = [25] 
learningRates   = [0.1] 
discounts       = [0.95]
explorDecayRate = [0.35]

# Setup experimental cases
stateFlags1 = [1,1,0,0,0,0]
stateFlags2 = [1,1,1,1,0,0]
stateFlags3 = [1,1,0,0,1,1]
stateFlags4 = [1,1,1,1,1,1]
stateFlags5 = [1,0,0,0,0,0]

stateSets   = [stateSetClass(stateFlags1, []), stateSetClass(stateFlags2, []), stateSetClass(stateFlags3, []), stateSetClass(stateFlags4, []), stateSetClass(stateFlags5, [])]

# Initialize environment and experiments
########################################################################

experimentDef = list(itertools.product(stateSets, statDivs, learningRates, discounts, explorDecayRate))
np.save("./{}/ExperimentDesign_{}.npy".format(path, path), experimentDef) 

# Setup and run each experiment
modFunction = filterProperties
experiments = []
for idx, experiment in enumerate(experimentDef): 

    # Setup experiment parameters 
    exp = expClass()
    exp.eType = "gridSearch"
    exp.eId = idx
    exp.name = "NA"
    exp.maxEps = 10000
    exp.solvedThr = 0
    exp.consecutiveWins = 10

    # stateParams
    sParams = stateSetClass(experiment[0].stateFlag, experiment[1])

    # Setup environment ranges for q-map
    env = envClass()
    env.env = gym.make("MountainCar-v0")

    envLow  = np.array(env.env.observation_space.low)
    envLow = [envLow[0],envLow[1],envLow[0],envLow[1],2*envLow[0],2*envLow[1]]        
    env.envLow = envLow

    envHigh = np.array(env.env.observation_space.high)
    envHigh = [envHigh[0],envHigh[1],envHigh[0],envHigh[1],2*envHigh[0],2*envHigh[1]]
    env.envHigh = envHigh

    # Setup q-learning parameters 
    qParams = qParamsClass()
    qParams.a = experiment[2]
    qParams.g = experiment[3]
    qParams.e = 0.95
    qParams.eDecay = experiment[4]
    qParams.qType = "map"

    # Setup render-learning 
    rParams = rParamsClass()
    rParams.render = False
    rParams.rendEvery = 500

    # Setup statistics parameters 
    statParams = statParamsClass()
    statParams.statsEvery = 2

    # Setup and run experiments sequential 
    # TODO: Extend to multithreading
    experiment = ql.qExperiment(exp, env, qParams, sParams, rParams, statParams, modFunction)
    experiment.run()
    experiment.saveResults(path)
    experiment.clear()





