# Dependencies
########################################################################
import numpy as np 
import pandas as pd 

# Dot indexing 
########################################################################
# http://localhost:8888/notebooks/mountainCar.ipynb
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Experiment Q-table data structures (QMap is q-table implemeted as a table)
# This approach removes the need to save memory of states that an agent
# will never encounter. 
########################################################################
class QMap: 

    # Name of state-space variables
    stateNames = []
    saMap = pd.DataFrame([]) 
    
    stateDivs  = [] # Number of bins for state space

    # Q-Map will still work if you exceed this range, though it is not recommended
    envLow     = [] # Minimum expected state range 
    envHigh    = [] # Maximum expected state range
    envRange   = [] # Range of expected state variabls
    
    def __init__(self, numActions, stateDivs, envLow, envHigh): 
        self.numActions = numActions
        self.stateDivs  = stateDivs         
        self.envLow     = envLow 
        self.envHigh    = envHigh 
        self.envRange   = envHigh - envLow         
        self.saMap      = pd.DataFrame([])        
        
    # get the discrete index from a continuous state variable 
    def stateIndex(self, state): 
        index = (state - self.envLow)/(self.envRange/(self.stateDivs))
        mapIdx = [str(i)+"." for i in tuple(index.astype(int))]
        strIdx = ''.join(mapIdx)
        return strIdx 
        
    # Add continuous state if it doesn't already exist in q-map 
    def addState(self, index): 
        if index not in self.saMap.columns:
            self.saMap[index] = np.random.uniform(low=2, high=0, size=self.numActions)
            print(index)
        return index
    
    # Return the maximum Q-value of a state action pair 
    def maxStateQ(self, state): 
        index = self.stateIndex(state)
        if index in self.saMap.columns:
            q = np.max(self.saMap[index])
            
        else: 
            self.addState(index)
            q = np.max(self.saMap[index])
            
        return q   
    
    # Return the action index associated the maximum Q-value for a given state 
    def maxQAction(self, state): 
        index = self.stateIndex(state)
        if index in self.saMap.columns:
            action = np.argmax(self.saMap[index])
        else: 
            self.addState(index)
            action = np.argmax(self.saMap[index])
        return action

    # Get the q value at a specified sate-action pair 
    def getQ(self, state, action): 
        index = self.stateIndex(state)
        if index not in self.saMap.columns:
            self.addState(index)
        return self.saMap[index][action]
                
    # Set the q value at a specified sate-action pair 
    def setQ(self, state, action, q): 
        index = self.stateIndex(state)
        if index not in self.saMap.columns:
            self.addState(index)
        self.saMap[index][action] = q     

    # Clear all parameters in the QMap 
    def clear(self): 
        self.stateNames = []
        self.saMap = pd.DataFrame([]) 
        
        self.stateDivs  = []
        self.envLow     = []
        self.envHigh    = []
        self.envRange   = []

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