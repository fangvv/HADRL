## HADRL

This is the source code for our paper: **Joint Service Caching and Computation Offloading in Mobile Edge Networks: A Hierarchical DRL Approach with Active Inference**. A brief introduction of this work is as follows:

> Mobile edge computing (MEC) is a promising paradigm that provides abundant computation and storage resources at the edge close to mobile devices (MDs). In MEC networks, MDs offload compute-heavy tasks to nearby edge servers (ESs) for delay-sensitive processing, where relevant services are stored to support task execution. However, the limited computation and storage capacities of ESs make joint optimization of service caching and computation offloading challenging due to coupled decisions, a large solution space, and dynamic environments. In this paper, we investigate the joint optimization of service caching and computation offloading in MEC networks, aiming to maximize the cache hit ratio and minimize the average service latency. To tackle this problem, the original formulation is decomposed into two hierarchical subproblems, namely high-level service caching and low-level computation offloading. We propose a novel hierarchical deep reinforcement learning (DRL) algorithm with active inference, termed HADRL. At the high-level, we adopt a deep deterministic policy gradient (DDPG) based DRL approach to maximize the cache hit ratio. At the low-level, we employ an active inference based DRL approach to minimize the average service latency. Unlike conventional DRL, the active inference based DRL approach selects policies by minimizing expected free energy instead of relying only on explicit rewards, making it well suited for highly dynamic low-level computation offloading. According to the simulation outcomes, the HADRL scheme surpasses the benchmark algorithms with respect to cache hit ratio as well as average service latency.

> 移动边缘计算（MEC）是一种前景广阔的计算范式，能够在靠近移动设备（MD）的网络边缘提供丰富的计算与存储资源。在MEC网络中，移动设备将计算密集型任务卸载到邻近的边缘服务器（ES）上进行时延敏感型处理，边缘服务器中需存储相关服务以支撑任务执行。然而，边缘服务器有限的计算与存储能力，使得服务缓存与计算卸载的联合优化面临决策相互耦合、解空间庞大以及环境动态变化等挑战。本文针对MEC网络中服务缓存与计算卸载的联合优化问题展开研究，旨在最大化缓存命中率并最小化平均服务时延。为解决该问题，我们将原始问题分解为两个分层子问题，即高层服务缓存与低层计算卸载。我们提出一种新颖的基于主动推理的分层深度强化学习（DRL）算法，称为HADRL。在高层，采用基于深度确定性策略梯度（DDPG）的DRL方法最大化缓存命中率；在低层，采用基于主动推理的DRL方法最小化平均服务时延。与传统DRL不同，基于主动推理的DRL方法通过最小化期望自由能来选择策略，而非仅依赖显式奖励信号，这使得其非常适合高度动态的低层计算卸载场景。仿真结果表明，HADRL方案在缓存命中率和平均服务时延方面均优于基准算法。

This work will be published by IEEE Internet of Things Journal. Click [here](https://doi.org/10.1109/JIOT.2026.3695862) for our paper.

## Required software

PyTorch 1.10.0

代码文件命名中ESX表示有该场景中有X个边缘服务器，默认2个。

## Citation

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

## Acknowledgement
To implement this repo, we refer to the code from [Active Inference](https://github.com/r-gould/active-inference). Thanks to [Rhys Gould](https://r-gould.github.io/2023/08/15/free-energy.html) for his great work.
	
## Contact

Haoyuan Li (24110127@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
