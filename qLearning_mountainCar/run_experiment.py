# Dependencies
########################################################################
import os 
import sys
sys.path.append(os.getcwd())
sys.path.append("./qLearning_mountainCar")

import os 
import numpy as np 
import gym
import itertools

from qLearning_mountainCar.models.q_learner import QLearner 
from qLearning_mountainCar.utils.utils import * 

# Sources
########################################################################
# Starting code:   https://www.youtube.com/watch?v=Gq1Azv_B4-4
# Example values:  https://github.com/svpino/lunar-lander
# Deep Q learning (TODO): https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf 

# Initialize training and learning parameters
########################################################################

def run_experimnts(path, statDivs, learningRates, discounts, explorDecayRate, stateSets): 

    # Initialize environment and experiments
    ########################################################################

    experimentDef = list(itertools.product(stateSets, statDivs, learningRates, discounts, explorDecayRate))
    np.save("./{}/ExperimentDesign.npy".format(path), experimentDef) 

    # Setup and run each experiment
    modFunction = filterProperties
    for idx, experiment in enumerate(experimentDef): 

        # Setup experiment parameters 
        exp                 = expClass()
        exp.eType           = "gridSearch"
        exp.eId             = idx
        exp.name            = "NA"
        exp.maxEps          = 10000
        exp.solvedThr       = 0
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
        experiment = QLearner(exp, env, qParams, sParams, rParams, statParams, modFunction)
        experiment.run()
        experiment.saveResults(path)
        experiment.clear()

if __name__ == "__main__": 

    # Details 
    path = "./tests/outputs"
    if not os.path.exists(path):
        os.makedirs(path)
   
    # Learning parameter options (every combination will be tested, so the experiment space can become very large)
    statDivs        = [  25] 
    learningRates   = [ 0.1] 
    discounts       = [0.95]
    explorDecayRate = [0.35]

    # Setup experimental cases
    stateFlags1 = [1,1,0,0,0,0]
    stateFlags2 = [1,1,1,1,0,0]
    stateFlags3 = [1,1,0,0,1,1]
    stateFlags4 = [1,1,1,1,1,1]
    stateFlags5 = [1,0,0,0,0,0]

    stateSets   = [stateSetClass(stateFlags1, []), stateSetClass(stateFlags2, []), stateSetClass(stateFlags3, []), stateSetClass(stateFlags4, []), stateSetClass(stateFlags5, [])]

    run_experimnts(path, statDivs, learningRates, discounts, explorDecayRate, stateSets)


