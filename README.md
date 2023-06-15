# Introduction: qLearning_mountainCar (^_^)
Project to run q_learning RL algorithm (w/ implementation) on the mountain car environment. Designed to test how agents learn better or worse when given information channels are turned on and off (x, y, dx, dy, ...). This project was completed as an assignment for the pattern recognition course hosted at the University of Florida, and was self-defined based upon general project requirements.

## Quick Start 

### Running the software
Run these commands (in the "qLearning_mountainCar" folder): 


```bash
$ conda create -n qLearning_mountainCar
$ conda activate qLearning_mountainCar
$ pip install -r requirements.txt
$ python qLearning_mountainCar\run_experiment.py
```

### Helpful classes
- QMap (in utils) is a graphical alternate to a q-table for q-learning that only stores the learned value for the visited states. This saves memory when the state space is large (expecially when the state space is descritized). 
- Classes/functions for running experiments (outdated - see x_printer from swormald)
- Functions for descritizing a continuous action space from gym 
- dotdict (way to access dict entries using dot indexing) 

## Unit Tests 
Run pytest with:
```bash
$ pytest tests
```

## Problem Statement
The purpose is to learn the basics of reinforcement learning via an implementationof Q-learning
to solve the Mountain Car game/environment made avaliable through the OpenAI gym. Please see the 
following resources for details of to the problem statement. 

- OpenAI_Gym:  https://gym.openai.com/
- MountainCar: https://gym.openai.com/envs/MountainCar-v0/

The purpose of this software was to test how changing the feedback modalities (agent inputs) would influence the
agent's ability to learn the value function or policy. It didn't - probably because the game is simple. 
The agent typically only has two state inputs from the environment, so I added 4 more.. see below. 

## (A few) Mountain Car Details

**Reward:**
Reward = -1 for all states, even when passing the goal. The reward is set to 0 via in this project when the car passes the objective position in the "env" object 

**Three actions:** 
0) Move Left
1) Do Nothing
2) Move Right

**Two (or more) states:** 
0) X position
1) X velocity
(ADDED - not native to the Mountain Car gym) 
2) X prior position
3) X prior velocity
4) dX distance difference
5) dX velocity difference

## Things that might be surprising: 
1) This does not use a q-table, but instead uses a "QMap" (self-defined class). This class is a table that helps to preserve memory as it doesn't not require you to have memory avaliable unless you have encountered a state before. Aside from that, it works the same as a q-table and has functions to help you get the values you want. 
2) There is an experiment class for q-learning to help setup and run a lot of tests quickly. It's not that hard to figure out. 
3) You need to make a folder for the software to put results or else it will crash. 

## Software Dependencies and Imports 
1) numpy  
2) gym
3) random 
4) matplotlib.pyplot
5) math 
6) pandas  
7) itertools
8) copy 

## Sources 
1) **Starting code:**   https://www.youtube.com/watch?v=Gq1Azv_B4-4
2) **Example values:**  https://github.com/svpino/lunar-lander
3) **Deep Q learning (TODO):** https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf 

