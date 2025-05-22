import os
import torch
import random, numpy as np
from pathlib import Path

import torch.amp

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=50000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 10000  # min. experiences before training
        self.learn_every = 1   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 20000   # no. of experiences between saving Mario Net
        self.save_dir = save_dir
        self.episode = 0

        self.use_cuda = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            if isinstance(state, list) and all(isinstance(x, np.ndarray) for x in state):
                state = np.array(state)
            state = np.array(state, dtype=np.float32)  # Convert list of arrays to a single array
            state = torch.from_numpy(state).cuda() if self.use_cuda else torch.from_numpy(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        self.memory.append((
            np.array(state, copy=False),
            np.array(next_state, copy=False),
            int(action),
            float(reward),
            bool(done)
        ))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        states, next_states, actions, rewards, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target):
        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if self.use_cuda else 'cpu'):
            loss = self.loss_fn(td_estimate, td_target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"

        # Convert all tensors in memory to CPU
        safe_memory = []
        for experience in self.memory:
            cpu_experience = tuple(
                x.detach().cpu() if isinstance(x, torch.Tensor) else x
                for x in experience
            )
            safe_memory.append(cpu_experience)

        torch.save(
            dict(
                online_model=self.net.online.state_dict(),
                target_model=self.net.target.state_dict(),
                optimizer=self.optimizer.state_dict(),
                exploration_rate=self.exploration_rate,
                curr_step=self.curr_step,
                episode=self.episode,
                memory=safe_memory
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found at {load_path}")

        checkpoint = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))

        self.net.online.load_state_dict(checkpoint["online_model"])
        self.net.target.load_state_dict(checkpoint["target_model"])
        # Load optimizer state
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        self.exploration_rate = checkpoint.get("exploration_rate", 1.0)
        self.curr_step = checkpoint.get("curr_step", 0)
        self.episode = checkpoint.get("episode", 0)

        if "memory" in checkpoint:
            self.memory = deque(checkpoint["memory"], maxlen=self.memory.maxlen)

        print(f"Loaded model from {load_path} at step {self.curr_step}, episode {self.episode}, exploration rate {self.exploration_rate}")
