# SF-TR
Task Relabelling for Multi-task Transfer using Successor Features

![](https://i.imgur.com/5clIoWY.png)

This repository contains the code for the paper "Task Relabelling for Multi-task Transfer using Successor Features". 

The ```agent``` contains the implementation of SFs. The Task Replacement is implemented as part of the loss function. Hindsight Task Replacement is done in the replay memory. 

The main entrypoint is ```main.py``` with various options. To reproduce the main SF results from the paper you may use the following options:

- ```--n-policies```: 1 to use a single one, -1 to use as many as features in the environment
- ```--replace-w```: Use Task Replacement (TR)
- ```--hindsight```: Use Hindsight Task Replacement (HTR)
- ```--train-mode```: Specifies the training setup
- ```--scenario```: Specifies the "reward function" used by the environment

The environment features various reward function ("scenarios") in the following categories:
- Pretrain: _pretrain_
- Stationary linear: _all_,  _one_item_, _two_item_
- Non-stationary linear: _random_, _random_pen_
- Stationary non-linear: _craft_staff_, _craft_sword_, _craft_bow_,  _craft_staff_neg_, _craft_sword_neg_, _craft_bow_neg_

In the pretrain setting the reward == 0 for all states.
When the reward function is stationary the same events give the same rewards for all episodes. When it is non-stationary a new reward function is sampled for each episode, i.e: collect a random resource.
Linear reward functions can be expressed as a linear combination of the events and their weights.
Some environments give negative rewards for collecting the wrong resources/items (_random_pen_, _craft_sword_neg_)

You may cite this project as:
```
@INPROCEEDINGS{balla2022relabelling,
  author={Balla, Martin and Perez-Liebana, Diego},
  booktitle={2022 IEEE Conference on Games (CoG)}, 
  title={Task Relabelling for Multi-task Transfer using Successor Features}, 
  year={2022},
  volume={},
  number={},
  pages={353-360},
  doi={10.1109/CoG51982.2022.9893550}}
```



