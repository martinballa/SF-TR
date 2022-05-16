from __future__ import division
import numpy as np
import torch
from utils.utils import preprocess_obs

def target_evaluate(args, env, T, agent):
  # Test performance over several episodes
  results = []
  ep_rewards = []
  total_rewards = 0

  done = True
  for episode in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False
        steps = 0
        task_rewards = 0
        phis = []

      state = preprocess_obs(state, args)
      if args.render:
        env.render()
      if args.scenario in ["random", "random_pen"]:
        task_vector = env.get_task()
      else:
        if args.train_mode == "target":
          task_vector = agent.w
        else:
          task_vector = env.get_task()
      action = agent.act(state, w=task_vector)
      state, reward, done, phi = env.step(action)
      reward_sum += reward
      phis.append(phi)
      steps+=1

      if torch.is_tensor(task_vector):
        task_vector = task_vector.detach().cpu().numpy()
      # calculate task reward based on observed features
      task_rewards += np.dot(phi, task_vector)

      if done:
        if torch.is_tensor(task_vector):
          task = task_vector.detach().cpu().numpy().round(2)
        else:
          task = task_vector.round(2)
        ep_rewards.append(reward_sum)
        results.append((T, steps, reward_sum, task_rewards, task, *np.sum(phis, axis=0)))
        total_rewards += reward_sum
        break

  avg_reward = total_rewards / args.evaluation_episodes
  print(f'Avg. reward: {np.mean(ep_rewards)} and std = {np.std(ep_rewards)} ')

  return results, avg_reward

def evaluate(args, env, T, agent, task=-1):
  # task_id may be used to specify which feature/sub-task we want to test
  # Test performance over several episodes
  results = []
  total_rewards = 0

  done = True
  for episode in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False
        steps = 0
        task_rewards = 0
        phis = []
        w = np.zeros(args.phi_dims)
        w[task] = 1

      state = preprocess_obs(state, args)
      if args.render:
        env.render()
      action = agent.act(state, w)

      state, reward, done, phi = env.step(action)
      task_reward = np.dot(w, phi)
      if task != -1:
        reward = task_reward
      task_rewards += task_reward
      reward_sum += reward
      phis.append(phi)
      steps+=1

      if done:
        results.append((T, steps, reward_sum, task_rewards, task, *np.sum(phis, axis=0)))
        total_rewards += reward_sum
        break

  avg_reward = total_rewards / args.evaluation_episodes

  return results, avg_reward
