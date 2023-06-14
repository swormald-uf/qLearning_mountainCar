import pytest
import torch
import qLearning_mountainCar.run_experiment as exp 
import qLearning_mountainCar.utils.utils as util 

import os 

def test_model():

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
    stateSets   = [exp.stateSetClass(stateFlags1, [])]

    exp.run_experimnts(path, statDivs, learningRates, discounts, explorDecayRate, stateSets)


