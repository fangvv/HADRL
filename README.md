# HADRL

[![GitHub](https://img.shields.io/badge/Project_Homepage-181717?logo=github)](https://github.com/fangvv/HADRL) — [https://github.com/fangvv/HADRL](https://github.com/fangvv/HADRL)

This is the source code for our paper: **Joint Service Caching and Computation Offloading in Mobile Edge Networks: A Hierarchical DRL Approach with Active Inference**. A brief introduction of this work is as follows:

Mobile edge computing (MEC) is a promising paradigm that provides abundant computation and storage resources at the edge close to mobile devices (MDs). In MEC networks, MDs offload compute-heavy tasks to nearby edge servers (ESs) for delay-sensitive processing, where relevant services are stored to support task execution. However, the limited computation and storage capacities of ESs make joint optimization of service caching and computation offloading challenging due to coupled decisions, a large solution space, and dynamic environments. In this paper, we investigate the joint optimization of service caching and computation offloading in MEC networks, aiming to maximize the cache hit ratio and minimize the average service latency. To tackle this problem, the original formulation is decomposed into two hierarchical subproblems, namely high-level service caching and low-level computation offloading. We propose a novel hierarchical deep reinforcement learning (DRL) algorithm with active inference, termed HADRL. At the high-level, we adopt a deep deterministic policy gradient (DDPG) based DRL approach to maximize the cache hit ratio. At the low-level, we employ an active inference based DRL approach to minimize the average service latency. Unlike conventional DRL, the active inference based DRL approach selects policies by minimizing expected free energy instead of relying only on explicit rewards, making it well suited for highly dynamic low-level computation offloading. According to the simulation outcomes, the HADRL scheme surpasses the benchmark algorithms with respect to cache hit ratio as well as average service latency.

移动边缘计算（MEC）是一种前景广阔的计算范式，能够在靠近移动设备（MD）的网络边缘提供丰富的计算与存储资源。在MEC网络中，移动设备将计算密集型任务卸载到邻近的边缘服务器（ES）上进行时延敏感型处理，边缘服务器中需存储相关服务以支撑任务执行。然而，边缘服务器有限的计算与存储能力，使得服务缓存与计算卸载的联合优化面临决策相互耦合、解空间庞大以及环境动态变化等挑战。本文针对MEC网络中服务缓存与计算卸载的联合优化问题展开研究，旨在最大化缓存命中率并最小化平均服务时延。为解决该问题，我们将原始问题分解为两个分层子问题，即高层服务缓存与低层计算卸载。我们提出一种新颖的基于主动推理的分层深度强化学习（DRL）算法，称为HADRL。在高层，采用基于深度确定性策略梯度（DDPG）的DRL方法最大化缓存命中率；在低层，采用基于主动推理的DRL方法最小化平均服务时延。与传统DRL不同，基于主动推理的DRL方法通过最小化期望自由能来选择策略，而非仅依赖显式奖励信号，这使得其非常适合高度动态的低层计算卸载场景。仿真结果表明，HADRL方案在缓存命中率和平均服务时延方面均优于基准算法。

This work will be published by IEEE Internet of Things Journal. Click [here](https://doi.org/10.1109/JIOT.2026.3695862) for our paper online.

## Required software

- PyTorch 1.10.0

- NumPy

- Matplotlib

> 代码文件命名中 ESX 表示该场景中有 X 个边缘服务器，默认 2 个。

## Project Structure

```
HADRL/
├── active_rl/ # Active inference modules (low-level policy)
│ ├── buffer.py # Replay buffer for the ensemble / reward model
│ ├── measures.py # Active-inference related measures
│ ├── models.py # Ensemble dynamics model & reward model
│ ├── normalizer.py # State / action normalizer
│ ├── planner.py # Cross-entropy method (CEM) planner
│ └── trainer.py # Trainer for ensemble + reward model
├── model_ddpg.py # DDPG agent (high-level service caching policy)
├── Replay_buffer.py # Replay buffer for DDPG
├── arguments.py # Hyper-parameters & runtime arguments
├── environment_ES2.py # MEC environment with 2 edge servers
├── environment_ES3.py # MEC environment with 3 edge servers
├── environment_ES4.py # MEC environment with 4 edge servers
├── environment_ES5.py # MEC environment with 5 edge servers
├── main_active_ES2.py # Main training script for the 2-ES scenario
├── main_active_ES3.py # Main training script for the 3-ES scenario
├── main_active_ES4.py # Main training script for the 4-ES scenario
├── main_active_ES5.py # Main training script for the 5-ES scenario
└── README.md
```

## Core Modules

### MEC Environment (`environment_ESX.py`)

The environment class that models the MEC system with multiple edge servers. The `ESX` suffix denotes the number of edge servers in the scenario (default `X = 2`). Key responsibilities include:

- Modeling the mobile devices (MDs), edge servers (ESs), wireless channels and computation / caching resources.
- Tracking the high-level service caching decisions made by the DDPG agent.
- Tracking the low-level computation offloading, bandwidth and resource allocation decisions made by the active inference agent.
- Computing the cache hit ratio and average service latency as the optimization objectives.
- Returning the observation, reward and termination signal at each MDP step.

> The state / action dimensions, number of MDs / ESs and other scenario parameters are configured in `environment_ESX.py` and can be tuned for different experimental settings.

### High-level DDPG Agent (`model_ddpg.py`)

The DDPG agent handles the **high-level service caching** decisions, aiming to maximize the cache hit ratio.

**Network architecture:**

| Network | Layers | Output Activation |
|---|---|---|
| Actor | `state_dim` → `num_units_1` → `num_units_2` → `action_dim` | tanh |
| Critic | `[state, action]` → `num_units_1` → `num_units_2` → `1` | linear (Q-value) |

**Key hyperparameters (default in `arguments.py`):**

| Parameter | Value | Description |
|---|---|---|
| `lr_a = 0.001` | Learning rate for Actor | |
| `lr_c = 0.002` | Learning rate for Critic | |
| `gamma = 0.95` | Discount factor | |
| `tau = 0.005` | Soft target update rate | |
| `batch_size = 64` | Training batch size | |
| `memory_size = 500000` | Experience replay buffer size | |
| `update_iteration = 10` | Number of gradient updates per step | |
| `num_units_1 = 256` | Number of units in MLP layer 1 | |
| `num_units_2 = 128` | Number of units in MLP layer 2 | |

**Key methods:**

- `select_action(state)` — Forward pass through the Actor network to produce a deterministic caching action.

- `update(args)` — Sample a batch from the replay buffer, compute the critic / actor losses, perform back-propagation and soft-update the target networks with rate `tau`.

- `save(args)` / `load(args)` — Persist / restore the Actor and Critic networks.

- `Replay_buffer.Replay_buffer` — First-in-first-out experience replay buffer with `push()` and `sample()` for DDPG training.

### Low-level Active Inference Agent (`active_rl/`)

The active inference agent handles the **low-level computation offloading** decisions, aiming to minimize the average service latency. Unlike conventional DRL, it selects policies by minimizing the expected free energy.

**Sub-modules:**

- `models.py` — Implements the `EnsembleModel` (an ensemble of probabilistic dynamics networks) and the `RewardModel`, which are used to predict the next state and reward.
- `buffer.py` — Stores the transitions used to train the ensemble / reward models.
- `normalizer.py` — Normalizes state and action values to stabilize the training of the ensemble networks.
- `planner.py` — Cross-Entropy Method (CEM) planner that rolls out candidate action sequences through the learned ensemble, scores them using expected free energy, and returns the best action sequence.
- `trainer.py` — Trains the ensemble dynamics model and the reward model on the replay buffer.
- `measures.py` — Helper functions implementing active-inference related measures (e.g., epistemic value, information gain).

**Key hyperparameters (default in `arguments.py`):**

| Parameter | Value | Description |
|---|---|---|
| `ensemble_size = 20` | Number of models in the dynamics ensemble | |
| `hidden_size = 400` | Hidden size of each ensemble network | |
| `plan_horizon = 20` | Length of the planned action sequence (CEM horizon) | |
| `optimisation_iters = 7` | Number of CEM iterations | |
| `n_candidates = 700` | Number of candidate action sequences per CEM iteration | |
| `n_train_epoch_large = 60` | Training epochs of the ensemble / reward model | |
| `top_candidates = 70` | Number of elite candidates kept after each CEM iteration | |
| `learning_rate = 1e-3` | Learning rate for the ensemble / reward optimizer | |
| `grad_clip_norm = 1000` | Gradient clipping norm for stable training | |
| `strategy = "information"` | Free-energy strategy used by the planner | |
| `use_reward = True` | Whether reward is included in the free energy | |
| `use_exploration = True` | Whether to add exploration noise to the planned action | |
| `expl_scale = 0.1` | Scale of the exploration noise | |
| `reward_scale = 1.0` | Scale of the reward term in the free energy | |

### Arguments (`arguments.py`)

Centralized configuration of all training, network and active-inference hyper-parameters. All scenarios (`ES2` – `ES5`) use the same set of arguments, with the environment file determined by the corresponding `main_active_ESX.py` entry script.

## Usage

```
# Install dependencies (PyTorch 1.10.0 required)
pip install torch==1.10.0 numpy matplotlib

# Run HADRL with 2 edge servers (default scenario)
python main_active_ES2.py

# Run HADRL with 3 / 4 / 5 edge servers
python main_active_ES3.py
python main_active_ES4.py
python main_active_ES5.py
```

## Citation

If you find HADRL useful or relevant to your project and research, please kindly cite our paper:

```
@ARTICLE{11534197,
    author={Li, Haoyuan and Lv, Zhenjie and Wang, Yuhang and He, Ying and Fang, Weiwei and Yu, Fei Richard},
    journal={IEEE Internet of Things Journal},
    title={Joint Service Caching and Computation Offloading in Mobile Edge Networks: A Hierarchical DRL Approach with Active Inference},
    year={2026},
    volume={},
    number={},
    pages={1-1},
    keywords={Algorithms;Optimization;Modeling;Resource management;Joints;Internet of Things;Timing;Learning (artificial intelligence);Convergence;Educational institutions;Mobile edge computation;computation offloading;service caching;deep reinforcement learning;active inference},
    doi={10.1109/JIOT.2026.3695862}
}
```

## Acknowledgement

To implement this repo, we refer to the code from [Active Inference](https://github.com/r-gould/active-inference). Thanks to [Rhys Gould](https://r-gould.github.io/2023/08/15/free-energy.html) for his great work.

## Contact

Haoyuan Li ([24110127@bjtu.edu.cn](mailto:24110127@bjtu.edu.cn))

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
