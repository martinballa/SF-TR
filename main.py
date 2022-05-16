import argparse
import os
import csv
import json
from datetime import datetime
import time

import numpy as np

from agent.sf import SF

from utils.replay_memory import ReplayMemory
from utils.utils import *
from utils.env import MiniCrafter
from utils.train_loops import *

parser = argparse.ArgumentParser(description='SF')
# SF arguments


# mainly SF args
parser.add_argument('--phi-dims', type=int, default=4, help='Dimension of the Successor Features')
parser.add_argument('--n-policies', type=int, default=1, help='Number of policies to use with SFs')
parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration (gets linearly reduced during training)')
parser.add_argument('--gamma', type=float, default=0.9, help='Discount rate')
parser.add_argument('--tau', type=float, default=0.01, help='Update proportion (used for soft target net updates)')
parser.add_argument('--rew-pred-weight', type=float, default=1.0, help='Reward prediction loss weight in update')
parser.add_argument('--soft-update', action='store_true', help='Do soft target network updates')
parser.add_argument('--train-mode', type=str, default='target', choices=['target', 'pretrain', 'transfer'], help='SF training method, defines which params are trained and which ones are fixed')
parser.add_argument('--replace-w', action='store_true', help='When updating psi loss replace original w to the policy\'s w')
parser.add_argument('--hindsight', action='store_true', help='When updating SF it applies Hindsight Task replacement on the sampled experience')


# Env args
parser.add_argument('--obs-type', type=str, default='obs', choices=["obs", "inv", "task", "full"], help="Observation type, observation only, obs+inventory or obs+inv+task")
parser.add_argument('--scenario', type=str, default='min', choices=["pretrain", "random", "random_pen", "craft_staff", "craft_sword", "craft_bow",  "craft_staff_neg", "craft_sword_neg", "craft_bow_neg", "all", "none", "one_item", "two_item", "min"], help="Reward scenario")

# common args
parser.add_argument('--gpu-id', type=int, default=0, help='Specify a GPU ID, mostly used when computer has multiple GPUs')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--seed', type=int, default=-1, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(1e6), metavar='STEPS', help='Number of agent-env interactions')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--framestack', type=int, default=1, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=64, metavar='SIZE', help='Network hidden size')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(2e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(1e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=20000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')

parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=1000000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')

# Setup
args = parser.parse_args()
if args.seed == -1:
  args.seed = np.random.randint(1, 10000)
  print(f"chosen seed = {args.seed}")
if args.train_mode == "pretrain" or args.scenario == "pretrain": # pretrain overwrites the scenario
  args.scenario = "pretrain"
  args.train_mode = "pretrain"

if args.scenario in ["random", "random_pen"]:
  args.obs_type = "full" # goal conditioned

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

# logging args
if args.model:
  args.model = os.path.expanduser(args.model)

args.logdir = os.path.expanduser(args.logdir)
date_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
exp_name = f"{args.scenario}/{args.seed}_{date_time}"
results_dir = os.path.join(args.logdir, exp_name) # , args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
if not args.evaluate:
  with open(f"{results_dir}/args.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)

# When evaluating only a sub-set of the arguments are used
if args.evaluate or args.train_mode == "transfer":
  json_file = os.path.join(os.path.dirname(args.model), "args.json")
  with open(json_file, "r") as f:
    exp_args = json.load(f)
  args_to_overwrite = ["train_mode", "gpu_id", "model", "evaluate", "render", "logdir", "evaluation_episodes"]
  for key in args.__dict__:
    if key in args_to_overwrite:
      exp_args[key] = args.__dict__[key]
  args.__dict__ = exp_args

# gpu setup
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda and args.gpu_id != -1:
  args.device = torch.device(f'cuda:{args.gpu_id}')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

# Environment setup env (training) eval_env (testing)
args.framestack = 1
args.hidden_dims = 64
args.phi_dims = len(MiniCrafter.get_feature_dict())
env = MiniCrafter(args)
action_space = env.action_space.n
eval_env = MiniCrafter(args)

if args.n_policies == -1:
  args.n_policies = args.phi_dims

# Logging related code
metrics = {'T': [], 'steps': [], 'rewards': [], 'mean_loss': [], 'task_rewards': [], 'task_id': []}
features = env.get_feature_dict()
metrics.update(features)
train_logger = ResultWriter(f"{results_dir}/train.csv", metrics=metrics)
args.phi_dims = len(features)
eval_metrics = {'T': [], 'steps': [], 'rewards': [], 'task_rewards': [], 'task_id': []}
eval_metrics.update(features)
eval_logger = ResultWriter(f"{results_dir}/eval.csv", metrics=eval_metrics)

# Agent
agent = SF(args, env)
mem = ReplayMemory(args=args)

if args.evaluate:
  eval_logger = ResultWriter(f"{results_dir}/test.csv", eval_metrics)
  if args.scenario == "pretrain":
    # evaluate agent on all possible sub-task
    for task in range(args.phi_dims):
      results, avg_reward = evaluate(args, eval_env, args.T_max, agent)
      eval_logger.write((results))
  else:
    results, avg_reward = target_evaluate(args, eval_env, args.T_max, agent)
    eval_logger.write((results))
  eval_logger.close()
elif args.train_mode == "transfer":
  training_scenario = args.scenario
  features = env.get_feature_dict()
  metrics.update(features)
  transfer_metrics = {"agent": [], "model": [], "n-policies": [], "replace_w": [], "hindsight": [], "source" : [], "target": [], "seed": [], 'T': [], 'steps': [], 'rewards': [], 'mean_loss': [], 'task_rewards': []}
  transfer_metrics.update(features)
  transfer_logger = ResultWriter(f"{args.logdir}/transfer.csv", metrics=transfer_metrics)

  if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
    # evaluate on all target envs
    targets = ["random", "random_pen", "one_item", "two_item", "craft_staff", "craft_sword", "craft_bow"]
    for t in targets:
      args.scenario = t
      transfer_env = MiniCrafter(args)
      results, avg_reward = target_evaluate(args, transfer_env, args.T_max, agent)
      transfer_logger.write([("SF", args.model, args.n_policies, args.replace_w, args.hindsight, training_scenario, t, args.seed, ) + (results[i]) for i in range(len(results))])


else:
  # main training loop calls
  if args.scenario == "pretrain":
    pretrain(args, agent, mem, env, train_logger, eval_env, eval_logger, results_dir)
  else:
    train_target(args, agent, mem, env, train_logger, eval_env, eval_logger, results_dir)


train_logger.close()
eval_logger.close()