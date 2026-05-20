# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np
import sys

class Buffer(object):
    def __init__(
        self,
        state_size,
        action_size,
        ensemble_size,
        normalizer,
        signal_noise=None,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.state_deltas = np.zeros((buffer_size, state_size))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state):
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        self._total_steps += 1
        
        self.normalizer.update(state, action, state_delta)


    def get_train_batches(self, batch_size, max_batches =1):
        size = len(self)
        #这个size是经验池中的数据
        # print("size",size)
        #indices是包含所有ensemble模型打乱的不同顺序的buffer数据 15个序列 每个序列长度为size
        #这个每次都是随机的 
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]

        # print("inde", len(indices))
        indices = np.stack(indices).T
        # print("inde",indices, len(indices))
        num_batches = min(max_batches, size // batch_size)


        for batch_num in range(num_batches):

            i = batch_num * batch_size

            j = i + batch_size

            # print("i , j ",i, j)
            batch_indices = indices[i:j]
            # print("1111",batch_indices)

            batch_indices = batch_indices.flatten()

            # print("len", len(batch_indices))
            # sys.exit()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)
            # print("statesss",states, states.shape, states[0])
            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(
                self.ensemble_size, batch_size, self.state_size
            )

            yield states, actions, rewards, state_deltas

            

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps
