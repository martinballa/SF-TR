from collections import deque
import torch
import numpy as np

class ReplayMemory():
    # keeps everything on GPU instead of manually shifting them back and forth
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.capacity = args.memory_capacity

        self.obs = deque(maxlen=args.memory_capacity)
        self.actions = deque(maxlen=args.memory_capacity)
        self.rewards = deque(maxlen=args.memory_capacity)
        self.dones = deque(maxlen=args.memory_capacity)
        self.features = deque(maxlen=args.memory_capacity)
        self.tasks = deque(maxlen=args.memory_capacity)
        self.dones = deque(maxlen=args.memory_capacity)
        self.invs = deque(maxlen=args.memory_capacity)

    def append(self, transition):
        """transition is a tuple of (s, a, s', r, done, psi, w)"""
        # moves observations to cpu
        transition = list(transition)

        if self.args.obs_type != "obs":
            obs = transition[0][0].detach().clone()
            inv = transition[0][1].detach().clone()
            self.invs.append(inv)
        else:
            obs = transition[0].detach().clone()

        self.obs.append(obs)
        self.actions.append(transition[1])
        self.rewards.append(transition[3])
        self.dones.append(transition[4])
        self.features.append(transition[5])
        self.tasks.append(transition[6])

    def _get_transitions(self, idx):
        # Collect all the transitions defined by a list of IDs idx
        transitions = []
        for id in idx:
            obs = []
            next_obs = []
            done = False
            if self.args.framestack > 1:
                for i in (np.arange(-self.args.framestack + id,  id)):
                    # fill up the buffer for framestack, append zeros when episode step < framestack
                    if self.dones[i]:
                        done = True
                        obs.append(torch.zeros_like(self.obs[i]))
                        next_obs.append(self.obs[i+1])
                    elif done:
                        obs.append(torch.zeros_like(self.obs[i]))
                        next_obs.append(torch.zeros_like(self.obs[i]))
                    else:
                        obs.append(self.obs[i])
                        next_obs.append(self.obs[i+1])
            else:
                # If we picked the final observation the agent did not act so we go a step back
                obs.append(self.obs[id])
                if self.dones[id] or id ==len(self.obs):
                    next_obs.append(torch.zeros_like(self.obs[id]))
                else:
                    next_obs.append(self.obs[id + 1])

            task = torch.tensor(self.tasks[id]).to(self.args.device)
            if self.args.hindsight:
                # see if any of the future exps observe features with non-zero entries before episode is over
                for i in range(id, len(self.tasks)):
                    if self.dones[i]:
                        # keep original task
                        task = torch.tensor(self.tasks[id]).to(self.args.device)
                        break
                    elif self.features[i].max() > 0:
                        # replace task by observed feature
                        task = torch.tensor(self.features[i]).to(self.args.device)
                        break

            extras = torch.tensor(self.features[id]), task
            if self.args.obs_type != "obs":
                transitions.append(
                    (torch.vstack(obs), self.actions[id], torch.vstack(next_obs),
                    self.rewards[id], self.dones[id], self.invs[id], self.invs[id+1], *extras))
            else:
                transitions.append((torch.vstack(obs), self.actions[id], torch.vstack(next_obs), self.rewards[id], self.dones[id], *extras))

        return transitions

    def sample(self, batch_size=32):
        # copies current batch and move to device
        # tuple is : s, a, s', r, done, w, phi
        idx = np.random.randint(0, len(self.rewards) -1, [batch_size])
        transition = list(zip(*self._get_transitions(idx)))

        if self.args.obs_type != "obs":
            transition[0] = torch.stack(transition[0]).detach().clone().to(self.device),\
                            torch.stack(transition[5]).detach().clone().float().to(self.device)
            transition[2] = torch.stack(transition[2]).detach().clone().to(self.device),\
                            torch.stack(transition[6]).detach().clone().float().to(self.device)
        else:
            transition[0] = torch.stack(transition[0]).detach().clone().to(self.device)
            transition[2] = torch.stack(transition[2]).detach().clone().to(self.device)
        transition[1] = torch.from_numpy(np.array(transition[1])).detach().clone().unsqueeze(1).to(self.device)
        transition[3] = torch.tensor(transition[3]).detach().clone().to(self.args.device)
        # done gets processed here, not in the update function
        transition[4] = torch.tensor((1 - np.stack(transition[4]))).detach().clone().to(self.device)
        transition[5] = torch.stack(transition[-2]).detach().clone().float().to(self.device)
        transition[6] = torch.stack(transition[-1]).detach().clone().float().to(self.device)
        return transition

    def reset(self):
        self.obs = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)
        self.features = deque(maxlen=self.capacity)
        self.tasks = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)
        self.invs = deque(maxlen=self.capacity)