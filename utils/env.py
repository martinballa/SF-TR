# modified Foraging World env based on aditimavalankar's implementation
import argparse
import os
import numpy as np
import gym
from gym import spaces, utils
import cv2
import torch

# Action keywords
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ITEM_TYPES = np.eye(5)

MAX_STEPS = 300
RES_COUNT = 3
TABLE_COUNT = 1
TRAP_COUNT = 10
NORM_INV = False # inventory normalization (divide by max + 1)

def get_initial_objects():
    return np.array([RES_COUNT, RES_COUNT, RES_COUNT, TABLE_COUNT, TRAP_COUNT], dtype=int)

class MiniCrafter(gym.Env, utils.EzPickle):
    def __init__(self, args,
                 grid_length=12,
                 n=5):
        # n: 0-2 resources, 3 crafting table, 4 trap
        self.args = args
        self.grid = np.zeros((grid_length, grid_length, n))
        self.resources = np.zeros(n)
        self.n = n
        self._scenario = args.scenario
        self.obs_type = args.obs_type

        self.observation_space = spaces.Box(low=-np.inf, high=1,
                                                shape=(n, grid_length, grid_length))
        self.action_space = spaces.Discrete(4)

        self.n_steps = 0
        self.initial_objects = get_initial_objects()
        # desirability is the "task"/w vector
        self._desirability = np.ones(self.n) if self._scenario == "all" else np.zeros(self.n)
        if self._scenario == "one_item":
            self._desirability = np.array([-1, 1, -1, 0, -1])
        elif self._scenario == "two_item":
            self._desirability = np.array([1, 1, -1, 0, -1])

        if self.args.render:
            from utils.env_renderer import Renderer
            self.renderer = Renderer(os.path.expanduser("./utils/assets/"))

    def reset(self):
        grid_length, _, _ = self.grid.shape
        self.grid = np.zeros((grid_length, grid_length, self.n))

        # Populate the grid.
        for index, n_items in enumerate(self.initial_objects):
            for _ in range(n_items):
                while True:
                    x, y = np.random.randint(grid_length, size=(2))
                    if (self.grid[x, y].sum() == 0 and
                            (x != int(grid_length / 2) or
                             y != int(grid_length / 2))):
                        break
                self.grid[x, y, index] = 1

        self.resources = np.zeros(self.n)
        self.n_steps = 0
        if self._scenario in ["random", "random_pen"]:
            self._desirability = self._sample_desirability()

        return self.get_observation()

    def step(self, a):
        self.n_steps += 1
        done = False if self.n_steps < MAX_STEPS else True
        self.update_grid(a)
        reward = 0

        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        item_type = np.zeros(self.n)

        min_resources = np.min(self.resources[:3])
        # if agent moves onto an object
        if self.grid[centre].sum() != 0:
            item_type = ITEM_TYPES[np.flatnonzero(self.grid[centre])[0]]
            # crafting
            if np.argmax(item_type) == 3:
                # check if agent can craft the target item
                if self._scenario == "craft_staff":
                    if self.resources[0] >= 1:
                        self.resources[0] -= 1
                        reward = 1
                elif self._scenario == "craft_sword":
                    if self.resources[0] >= 1 and self.resources[1] >= 1:
                        self.resources[0] -= 1
                        self.resources[1] -= 1
                        reward = 1
                elif self._scenario == "craft_bow":
                    if min_resources >= 1:
                        self.resources += np.array([-1, -1, -1, 0, 0])
                        reward = 1
                elif self._scenario == "craft_staff_neg":
                    if self.resources[0] >= 1:
                        self.resources[0] -= 1
                        item_type[0] = -1
                        reward = 1
                elif self._scenario == "craft_sword_neg":
                    if self.resources[0] >= 1 and self.resources[1] >= 1:
                        self.resources[0] -= 1
                        self.resources[1] -= 1
                        item_type[0] = -1
                        item_type[1] = -1
                        reward = 1
                elif self._scenario == "craft_bow_neg":
                    if min_resources >= 1:
                        self.resources += np.array([-1, -1, -1, 0, 0])
                        item_type = np.array([-1, -1, -1, 1, 0])
                        reward = 1
                self.spawn_new_item(np.flatnonzero(self.grid[centre])[0])
                self.grid[centre] = np.zeros(self.n)
            # trap
            elif np.argmax(item_type) == 4:
                done = True
                reward = -1
            else:
                self.resources += item_type
                self.spawn_new_item(np.flatnonzero(self.grid[centre])[0])
                self.grid[centre] = np.zeros(self.n)
                reward = np.sum(item_type * self._desirability)

        if self.args.render:
            self.render()

        return self.get_observation(), reward, done, np.array(item_type)

    def get_observation(self):
        obs = self.grid.copy()
        grid_length = obs.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        obs[centre] = np.ones(self.n)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        obs = obs.permute(2, 0, 1)
        if self.obs_type == "inv":
            inv = torch.tensor(self.resources.copy(), dtype=torch.float32, device=self.args.device)
            if NORM_INV: inv = inv/(max(inv) + 1)
            return obs, inv
        elif self.obs_type == "task":
            return obs, torch.tensor(self._desirability.copy(), dtype=torch.float32, device=self.args.device)
        elif self.obs_type == "full":
            inv = self.resources.copy()
            if NORM_INV: inv = inv/(max(inv) + 1)
            return obs, torch.tensor(np.concatenate((inv, self._desirability.copy())), dtype=torch.float32, device=self.args.device)
        else:
            return obs

    def get_task(self):
        # returns the hand-crafted task vector
        if self._scenario == "craft_staff":
            return np.array([0.5, 0, 0, 1, -1])
        elif self._scenario == "craft_sword":
            return np.array([0.5, 0.5, 0, 1, -1])
        elif self._scenario == "craft_bow":
            return np.array([0.5, 0.5, 0.5, 1, -1])
        return self._desirability.copy()

    @staticmethod
    def get_feature_dict():
        features = {}
        items = ["wood", "iron", "string", "table", "trap"]
        for item in items:
            features.update({item: 0})
        return features

    def get_scalar_dims(self):
        if self.obs_type in ["inv", "task"]:
            return self.n
        elif self.obs_type == "full":
            return self.n * 2
        else:
            return 0

    def render(self):
        # todo the problem is that the channel at index 0 disappears due to taking the argmax
        AGENT = 6
        obs = self.grid.copy()
        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        # todo argmax
        obs = np.argmax(obs, axis=-1)
        obs[centre] = AGENT
        self.renderer.render(obs)

        self.grid[centre] = np.zeros(self.n)

    def get_obs(self):
        grid_copy = self.grid.copy()
        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        grid_copy[centre] = np.ones(self.n)
        scale = max(int(400 / grid_copy.shape[1]), 1)
        modified_size = grid_copy.shape[1] * scale
        img = cv2.resize(grid_copy, (modified_size, modified_size),
                         interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0.375, 0.375, 0.375])
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        return img

    def close(self):
        cv2.destroyAllWindows()

    def _sample_desirability(self):
        id = np.random.randint(3)
        multiplier = 0 if self._scenario == "random" else -1
        w = np.ones(self.n) * multiplier
        # w[3:] = 0 # remove non-resources from the desirability vector
        w[id] = 1
        return w

    def update_grid(self, a):
        if a == UP:
            self.grid = np.concatenate((self.grid[-1:, :, :],
                                        self.grid[:-1, :, :]),
                                       axis=0)

        elif a == DOWN:
            self.grid = np.concatenate((self.grid[1:, :, :],
                                        self.grid[:1, :, :]),
                                       axis=0)

        elif a == RIGHT:
            self.grid = np.concatenate((self.grid[:, 1:, :],
                                        self.grid[:, :1, :]),
                                       axis=1)

        elif a == LEFT:
            self.grid = np.concatenate((self.grid[:, -1:, :],
                                        self.grid[:, :-1, :]),
                                       axis=1)


    def spawn_new_item(self, item_type_index):
        grid_length = self.grid.shape[0]
        while True:
            x, y = np.random.randint(grid_length, size=(2))
            if (self.grid[x, y].sum() == 0 and (x != int(grid_length / 2) or
                                                y != int(grid_length / 2))):
                break
        self.grid[x, y, item_type_index] = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MiniCrafter')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--obs-type', type=str, default='obs', choices=["obs", "inv", "task", "full"],
                        help="Observation type, observation only, obs+inventory or obs+inv+task")
    args = parser.parse_args()
    args.scenario = "pretrain"
    args.train_mode = "pretrain"
    args.device = torch.device('cpu')

    env = MiniCrafter(args)
    state, done = env.reset(), False
    steps, rewards = 0, 0
    while not done:
        state, reward, done, phi = env.step(env.action_space.sample())
        steps+=1
        rewards += reward
        # env.render()
        if done:
            print(f"Total Rewards: {rewards} collected in {steps}")
            state, done = env.reset(), False
