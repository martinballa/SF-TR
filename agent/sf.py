import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger

class DQN(nn.Module):
    def __init__(self, args, input_dims, n_actions, scalar_dims=-1):
        super(DQN, self).__init__()
        self.args = args
        self.n_actions = n_actions
        self.framestack = args.framestack
        self.input_dims = input_dims
        self.hidden_units = args.hidden_size
        self.phi_dims = args.phi_dims
        self.n_policies = args.n_policies
        self.scalar_dims = scalar_dims
        self.device = args.device

        self.convs = nn.Sequential(nn.Conv2d(input_dims[0] * self.args.framestack, 8, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            self.conv_out_size = self.convs(torch.zeros(1, *self.input_dims)).flatten().shape[0]
        self.phi = nn.Linear(self.conv_out_size, self.hidden_units)

        # get SF psi
        self.psi = nn.Sequential(nn.Linear(self.hidden_units, self.hidden_units), nn.ReLU(),
                                 nn.Linear(self.hidden_units, self.phi_dims * self.n_actions * self.n_policies))

    # immediate reward prediction \phi \dot w
    def r_(self, phi, w):
        if len(phi.shape) == 1:
            phi = phi.unsqueeze(0)
        r = torch.mm(phi, w.unsqueeze(-1))
        return r.squeeze()

    # predict psi(z) (SF) for all actions
    def psi_(self, z):
        psi = self.psi(z)
        batch_size = z.shape[0]
        return psi.view(batch_size, self.n_policies, self.n_actions, -1)

    def psi_a_(self, psi, a, o):
        psi = psi[:, o]
        psi_a_ = psi.gather(1, a.repeat(1, self.phi_dims).unsqueeze(1)).squeeze(1)
        return psi_a_

    def q_(self, psi, w, policy=None):
        # Uses GPI + GPE if option is not provided otherwise acts using the provided option
        if not torch.is_tensor(w):
            w = torch.from_numpy(w).to(self.device)
        if len(w.shape) == 1:
            w = w.unsqueeze(0).unsqueeze(0)
        else:
            w = w.unsqueeze(1)
        if policy is not None:
            return torch.mul(psi[:, policy], w).sum(-1)
        else:
            return torch.mul(psi, w).sum(-1).max(1).values

    def get_sf(self, state):
        phi = self.encode(state)
        psi = self.psi_(phi)
        return psi

    # reward is phi \dot weight
    def get_r(self, phi, w):
        r = self.r_(phi, w)
        return r

    def encode(self, x):
        if self.args.obs_type == "obs":
            obs = x
        else:
            obs, inv = x[0], x[1]
        conv = self.convs(obs)
        x = conv.view(conv.shape[0], -1)
        if self.scalar_dims > 0:
            # process inv
            x_ = F.relu(self.scalar_input(inv))
            x_ = F.relu(self.scalar_fc(x_))

            # concat, predict
            x = torch.cat((x, x_), axis=1)

        z = self.phi(x)
        return z

    # Full path with all the heads used working from the state
    def forward(self, x, policy=None, w=None):
        # if option == None then it uses GPI+GPE
        z = self.encode(x)  # z is phi "the state feature encoding"
        psi = self.psi_(z) # psi has the shape [BS, policy, N_actions, Feature dims]

        q = self.q_(psi, w=w, policy=policy)
        return q, psi


class SF():
    def __init__(self, args, env):
        self.name = "SF"
        self.args = args
        self.device = args.device
        self.phi_dims = args.phi_dims
        self.n_policies = args.n_policies
        self.framestack = args.framestack

        self.env = env
        self.n_actions = env.action_space.n
        self.input_dims = env.observation_space.shape

        self.random = np.random.RandomState(args.seed)
        torch.manual_seed(args.seed)

        self.q_net = DQN(args, self.input_dims, self.n_actions).to(args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                checkpoint = torch.load(args.model,
                                        map_location='cpu')
                self.q_net.load_state_dict(checkpoint['state_dict'])
                print("Loading trained model: " + args.model)
            else:
                raise FileNotFoundError(args.model)
        self.target_net = DQN(args, self.input_dims, self.n_actions).to(args.device)
        self.update_target_net()

        params = list(self.q_net.parameters())

        self.batch_size = args.batch_size
        self.train_mode = args.train_mode

        # Depending on training mode different params are optimised
        if self.train_mode == "transfer":
            self.w = torch.zeros(args.phi_dims, requires_grad=True, device=args.device)
            params = []
            params.append(self.w)
        elif self.train_mode == "pretrain":
            w = torch.ones(args.phi_dims, requires_grad=True, device=args.device)
            self.set_w(w)
        elif self.train_mode == "target":
            self.w = torch.zeros(args.phi_dims, requires_grad=True, device=args.device)
            params.append(self.w)
        else:
            w = torch.ones(args.phi_dims)
            self.set_w(w)

        self.optim = torch.optim.Adam(params, lr=args.learning_rate)

        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.tau = args.tau
        self.norm_clip = args.norm_clip

        self.logger = Logger(headers=["Rewards", "Steps", "Success", "loss", "Reward Pred. loss"])
        self.test_logger = Logger(headers=["Rewards", "Steps", "Success"])
        self.soft_update = args.soft_update

        self.update_counter = 0
        self.ep_r_losses = []

        self.ep_q_values = []
        self.q_actions = np.zeros(self.n_actions)

    def set_w(self, w):
        self.w = w.clone().detach()

    def act(self, state, w=None, policy=None, epsilon=0.0):
        if self.random.uniform(0, 1) < epsilon:
            action = self.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                if w is not None:
                    q = self.q_net(state, policy=policy, w=w)[0]
                else:
                    q = self.q_net(state, policy=policy, w=self.w)[0]
                self.ep_q_values.append(q.squeeze().detach())
                action = q.argmax().item()
                self.q_actions[action] += 1
        return action

    def get_q(self, state, w=None, policy=None):
        state = state.unsqueeze(0).float().div(255).to(self.device)
        with torch.no_grad():
            return self.q_net(state, w=w, policy=policy)[0].squeeze()

    def r_loss(self, transitions):
        # Calculates the loss on w
        r = transitions[3]
        phi = transitions[5]

        r_ = self.q_net.get_r(phi, self.w)
        loss = F.mse_loss(r_.squeeze(), r.float())

        return self.args.rew_pred_weight * loss

    # PSI loss
    def psi_loss(self, transitions):
        # loss is the expected phi(s) + psi(s', a') - psi(s, a) where a' is max Q(s', a')

        latent = self.q_net.encode(transitions[0])
        psi = self.q_net.psi_(latent)

        loss = torch.zeros(1).to(self.device)

        # in loss the same features are used but the whole thing is recalculated for each policy
        for policy in range(self.n_policies):
            psi_a = self.q_net.psi_a_(psi, transitions[1], policy)

            with torch.no_grad():
                # use option as task
                if self.args.replace_w:
                    w = F.one_hot(torch.tensor(policy).repeat(self.batch_size), num_classes=self.phi_dims).to(
                        self.device)
                else:
                    w = transitions[6]

                # select argmax Q(s',a')
                latent_ = self.target_net.encode(transitions[2])
                psi_ = self.target_net.psi_(latent_)
                q_ = self.target_net.q_(psi_, w=w, policy=policy)
                best_a_ = q_.squeeze().argmax(1).unsqueeze(-1)
                psi_a_ = self.target_net.psi_a_(psi_, best_a_, policy)

            target = transitions[5] + (self.gamma * psi_a_ * transitions[4].unsqueeze(-1).repeat(1, self.args.phi_dims))
            loss += F.smooth_l1_loss(psi_a, target)
        return loss

    def learn(self, memory):
        # update network
        experience = memory.sample(self.batch_size)

        loss = self.psi_loss(experience)
        if self.train_mode in ["target", "transfer"]:
            loss += self.r_loss(transitions=experience)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.norm_clip)
        self.optim.step()

        self.update_counter += 1

        # Move the params of the target net towards q_net
        if self.soft_update:
            self.soft_update_target_net(self.tau)
        elif self.update_counter % 1000 == 0:
            self.update_target_net()
        return loss.detach().cpu().numpy()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update_target_net(self, tau):
        # moves the target net toward q_net
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)

    def save_model(self, path, name="checkpoint.pth"):
        state = {
            "state_dict": self.q_net.state_dict(),
            "optimizer": self.optim.state_dict(),
            "args": self.args,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "norm_clip": self.norm_clip,
            "w": self.w,
        }
        torch.save(state, os.path.join(path, name))

    def load_model(self, path, eval_only=False):
        state = torch.load(path)
        self.config = state["config"]
        self.q_net = DQN(self.config)
        self.q_net.load_state_dict(state["state_dict"])
        self.w = state["w"]

        if eval_only:
            self.q_net.eval()
        else:
            # load and copy q_net params to target_net
            self.target_net = DQN(self.config)
            self.update_target_net()
            self.optim.load_state_dict(state["optimizer"])
            self.epsilon = state["epsilon"]
            self.gamma = state["gamma"]
            self.norm_clip = state["norm_clip"]

    def train(self):
        self.q_net.train()

    def eval(self):
        self.q_net.eval()