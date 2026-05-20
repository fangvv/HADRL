# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, act_fn="swish"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.act_fn_name = act_fn
        self.act_fn = self._get_act_fn(self.act_fn_name)
        self.reset_parameters()

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, act_fn_name):
        if act_fn_name == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn_name == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn_name):
        if act_fn_name == "swish":
            return swish
        elif act_fn_name == "linear":
            return lambda x: x


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ensemble_size,
        normalizer,
        act_fn="swish",
        device="cpu",
    ):
        super().__init__()

        self.fc_1 = EnsembleDenseLayer(
            in_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_2 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_3 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_4 = EnsembleDenseLayer(
            hidden_size, out_size * 2, ensemble_size, act_fn="linear"
        )

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.device = device
        self.to(device)

    def forward(self, states, actions):
        norm_states, norm_actions = self._pre_process_model_inputs(states, actions)
        norm_delta_mean, norm_delta_var = self._propagate_network(
            norm_states, norm_actions
        )
        delta_mean, delta_var = self._post_process_model_outputs(
            norm_delta_mean, norm_delta_var
        )
        return delta_mean, delta_var

    def loss(self, states, actions, state_deltas):
        states, actions = self._pre_process_model_inputs(states, actions)
        delta_targets = self._pre_process_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    # def sample(self, mean, var):
    #     return Normal(mean, torch.sqrt(var)).sample()

    def sample(self, mean, var):
        #因为var方差后边为0，所以这里要加一个小的值
        # var = torch.clamp(var, min=1e-6)
        var = torch.where(
            torch.isfinite(var),
            var,
            torch.tensor(1e-6, device=var.device, dtype=var.dtype)
        )
        var = torch.clamp(var, min=1e-6)

        std = torch.sqrt(var)
        std = torch.where(
            torch.isfinite(std),
            std,
            torch.tensor(1e-3, device=std.device, dtype=std.dtype)
        )

        mean = torch.where(
            torch.isfinite(mean),
            mean,
            torch.zeros_like(mean)
        )
        return Normal(mean, std).sample()


    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.fc_3(op)
        op = self.fc_4(op)

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (
            self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        )
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, delta_var):
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var



class RewardModel(nn.Module):
    def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)
        self.reset_parameters()
        self.to(device)

    def forward(self, states, actions):
        inp = torch.cat((states, actions), dim=-1)
        reward = self.act_fn(self.fc_1(inp))
        reward = self.act_fn(self.fc_2(reward))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, actions, rewards):
        r_hat = self(states, actions)
        return F.mse_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.to(self.device)





# class RewardModel(nn.Module):
#     def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
#         super().__init__()
#         self.in_size = in_size
#         self.hidden_size = hidden_size
#         self.device = device
#         # self.act_fn = getattr(F, act_fn)
#         self.act_fn = nn.LeakyReLU(0.01)
#         self.act_fn_name = "leaky_relu"
#         self.tanh = nn.Tanh()
#         self.reset_parameters()
#         self.to(device)

#     def forward(self, states, actions):
#         inp = torch.cat((states, actions), dim=-1)
#         reward = self.act_fn(self.fc_1(inp))
#         reward = self.act_fn(self.fc_2(reward))
#         reward = self.tanh(self.fc_3(reward).squeeze(dim=1))
#         return reward

#     def loss(self, states, actions, rewards):
#         r_hat = self(states, actions)
#         return F.mse_loss(r_hat, rewards)

#     def reset_parameters(self):
#         self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
#         self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc_3 = nn.Linear(self.hidden_size, 1)

#         gain = nn.init.calculate_gain(self.act_fn_name) if self.act_fn_name != "swish" else 1.0
#         nn.init.kaiming_normal_(self.fc_1.weight, nonlinearity=self.act_fn_name)
#         nn.init.kaiming_normal_(self.fc_2.weight, nonlinearity=self.act_fn_name)
#         nn.init.xavier_normal_(self.fc_3.weight, gain=1.0)  # Linear output layer
#         nn.init.constant_(self.fc_1.bias, 0.0)
#         nn.init.constant_(self.fc_2.bias, 0.0)
#         nn.init.constant_(self.fc_3.bias, 0.0)
#         self.to(self.device)









# class RewardModel(nn.Module):
#     def __init__(self, in_size, hidden_size_1=128, hidden_size_2=128, act_fn="relu", dropout_rate=0.1, device="cpu"):
#         super(RewardModel, self).__init__()
#         self.in_size = in_size
#         self.hidden_size_1 = hidden_size_1
#         self.hidden_size_2 = hidden_size_2
#         self.device = torch.device(device)
#         self.dropout_rate = dropout_rate

#         # Define layers
#         self.fc_1 = nn.Linear(in_size, hidden_size_1)
#         self.fc_2 = nn.Linear(hidden_size_1, hidden_size_2)
#         self.fc_3 = nn.Linear(hidden_size_2, 1)
#         self.bn_1 = nn.BatchNorm1d(hidden_size_1)
#         self.bn_2 = nn.BatchNorm1d(hidden_size_2)
#         self.dropout = nn.Dropout(dropout_rate)

#         # Define activation function
#         self.act_fn_name = act_fn.lower()
#         if self.act_fn_name == "swish":
#             self.act_fn = lambda x: x * torch.sigmoid(x)
#         elif self.act_fn_name in ["relu", "leaky_relu", "tanh"]:
#             self.act_fn = getattr(nn, self.act_fn_name.capitalize())()
#         else:
#             raise ValueError(f"Unsupported activation function: {act_fn}")

#         # Initialize parameters and move to device
#         self.reset_parameters()
#         self.to(self.device)

#     def reset_parameters(self):
#         """Initialize weights and biases with appropriate methods."""
#         gain = nn.init.calculate_gain(self.act_fn_name) if self.act_fn_name != "swish" else 1.0
#         nn.init.kaiming_normal_(self.fc_1.weight, nonlinearity=self.act_fn_name)
#         nn.init.kaiming_normal_(self.fc_2.weight, nonlinearity=self.act_fn_name)
#         nn.init.xavier_normal_(self.fc_3.weight, gain=1.0)  # Linear output layer
#         nn.init.constant_(self.fc_1.bias, 0.0)
#         nn.init.constant_(self.fc_2.bias, 0.0)
#         nn.init.constant_(self.fc_3.bias, 0.0)

#     def forward(self, states, actions):
#         """Forward pass to predict reward."""
#         # Input validation
#         if states.dim() == 1:
#             states = states.unsqueeze(0)
#         if actions.dim() == 1:
#             actions = actions.unsqueeze(0)
#         if states.size(-1) + actions.size(-1) != self.in_size:
#             raise ValueError(f"Expected input size {self.in_size}, got {states.size(-1) + actions.size(-1)}")

#         # Ensure inputs are on the correct device
#         states, actions = states.to(self.device), actions.to(self.device)

#         # Forward pass
#         inp = torch.cat((states, actions), dim=-1)
#         x = self.act_fn(self.bn_1(self.fc_1(inp)))
#         x = self.dropout(x)
#         x = self.act_fn(self.bn_2(self.fc_2(x)))
#         x = self.dropout(x)
#         reward = self.fc_3(x).squeeze(dim=-1)
#         return reward

#     def loss(self, states, actions, rewards, loss_fn="mse"):
#         """Compute loss between predicted and true rewards."""
#         rewards = rewards.to(self.device)
#         if rewards.dim() == 1:
#             rewards = rewards.unsqueeze(-1)
#         r_hat = self(states, actions)
#         if loss_fn.lower() == "mse":
#             return F.mse_loss(r_hat, rewards.squeeze(-1))
#         elif loss_fn.lower() == "l1":
#             return F.l1_loss(r_hat, rewards.squeeze(-1))
#         else:
#             raise ValueError(f"Unsupported loss function: {loss_fn}")