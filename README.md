# SF-TR
Task Relabelling for Multi-task Transfer using Successor Features

This repository contains the code for the paper "Task Relabelling for Multi-task Transfer using Successor Features". 

The ```agent``` contains the implementation of SFs. The Task Replacement is implemented as part of the loss function. Hindsight Task Replacement is done in the replay memory. 

The main entrypoint is ```main.py``` with various options. To reproduce the main SF results from the paper you can use the following options:

- ```--n-policies```: 1 to use a single one, -1 to use as many as features in the environment
- ```--replace-w```: Use Task Replacement (TR)
- ```--hindsight```: Use Hindsight Task Replacement (HTR)
- ```--train-mode```: Specifies the training setup
- ```--scenario```: Specifies the "reward function" used by the environment




