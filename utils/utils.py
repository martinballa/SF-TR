import os
import torch
import csv

class ResultWriter():
    # Logging related code
    def __init__(self, results_file, metrics):
        self.metrics = metrics

        # self.results = []
        self.results_file = open(results_file, "a")
        self.result_writer = csv.writer(self.results_file)
        if os.stat(results_file).st_size==0:
            self.result_writer.writerow(self.metrics.keys())

    def write(self, results):
        self.result_writer.writerows((results))
        self.results_file.flush()

    def close(self):
        self.results_file.close()

def preprocess_obs(state, args):
    # If needed this is the place to modify the preprocessing
    if args.obs_type != "obs":
        state, scalars = state
    state = state.clone()
    if len(state.shape) == 3:
        state = torch.unsqueeze(state, 0)
    state = state.float()
    if args.obs_type != "obs":
        return state, scalars.unsqueeze(0)
    return state

def get_epsilon(episode, max_episodes, final_epsilon=0.0):
  # linearly decay towards 0
  lr = ((max_episodes - episode) / max_episodes)  # * 1 - final_epsilon
  if (lr < final_epsilon):
    lr = final_epsilon
  return lr
