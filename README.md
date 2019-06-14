# DRLND DQN Banana Navigation Project  
---
This is the first project in the Udacity Deep Reinforcement Learning Nanodegree. It requires to develop and train a [Deep Q-Network (DQN)](https://deepmind.com/research/dqn/) model to collect yellow bananas in a simulator. 

[![Trained DQN Agent](https://github.com/hortovanyi/DRLND-DQN-Banana-Navigation/blob/master/output/agent_run_writeup.gif?raw=true)](https://www.youtube.com/watch?v=adfIUz7Ex5g)

The Agent is incentivised to find Yellow bananas with +1 reward and disincentivised to find Blue bananas with -1 reward within the simulator.


## Project Details
The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 

The environment is considered solved when the average reward (over the last 100 episodes) is at least 13+.

For detialed information about this project execution and future work , please go through this  <a href="Report.md"> Report</a>

## Getting Started
It is recommended to follow the Udacity DRL ND dependencies [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies) 

This project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 

A prebuilt simulator is required in be installed. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped.

Next, before starting the environment utilising the corresponding prebuilt app from Udacity  **_Before running the code cell in the notebbok**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Banana.app"`
- **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
- **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
- **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
- **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
- **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
- **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Banana.app")
```

## Instructions

For training the agent , Please follow the instructions in  <a href="NavigationCompletedDQN.ipynb"> NavigationCompletedDQN </a> from the begining.

However, to just checkout the already trained agent's performance. you should run the following script.

