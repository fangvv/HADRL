# time 20240605
# by qian
# 利用twin-actor ddpg 在大时间尺度上做出缓存，在小时间尺度上做出带宽、计算资源分配和计算卸载的决策！！
# 用做一个算法对比
# 这里先做两个EDGE SERVER 的情况
import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from environment_ES3 import MultiAgentEnv
from model_ddpg import DDPG1
from arguments import parse_args



# from active_rl.buffer import Buffer
# from active_rl.models import EnsembleModel, RewardModel
# from active_rl.planner import Planner
# from active_rl.normalizer import Normalizer
# from active_rl.trainer import Trainer

from new_active_rl.dataset import Dataset
from new_active_rl.inference import inference
from new_active_rl.transition import Transition
from new_active_rl.reward import Reward
from new_active_rl.normalizer import Normalizer
from new_active_rl.ensemble import Ensemble
from new_active_rl.trainer import Trainer


import sys



def train(arglist):


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # train the agents in the offload environment
    # input arglist: env parameters
    # out the reward of agents , the model needed to save
    print("""step1: create the environment """)
    env = MultiAgentEnv()
    print('step 1 Env {} is right ...'.format(arglist.scenario_name))

    print("""step2: create agents""")
    # 首先是获取到 服务器智能体的状态和动作的维度，用来产生智能体：
    # print("env.observation_space_server_cache",env.observation_space_server_cache[0])
    # 这的状态空间，假定大小时间尺度一样，都包含环境信息和服务缓存的状态，
    obs_dim = env.observation_space_server_cache[0].shape[0]

    obs_dim_small = env.observation_space_samll[0].shape[0]
    # print("obs_dim_small",obs_dim_small)
    # print('obs_dim',obs_dim)
    # large-timescale caching dimation
    action_large_dim = env.action_space_server_cache[0].shape[0]
    # small-timescale offloading, bandwidth , frequency
    action_small_dim = env.num_User * 3 
    print("action_small_dim",action_small_dim)
    # ddpg
    # twin_actor_agent = DDPG(obs_dim,action_small_dim, action_large_dim, arglist)



    twin_actor_agent_large = DDPG1(obs_dim,action_large_dim, arglist)
    twin_actor_agent_small = DDPG1(obs_dim_small, action_small_dim, arglist)




    # 定义新主动推理 1------------------------------------------------------------- #
    normalizer = Normalizer()
    transition = Transition(arglist.ensemble_size, 
                        node_info=[obs_dim_small+action_small_dim, arglist.hidden_size, arglist.hidden_size, arglist.hidden_size, 2*obs_dim_small],
                        normalizer=normalizer)
    transition.to(device)

    reward = Reward(node_info=[obs_dim_small+action_small_dim, arglist.hidden_size, arglist.hidden_size, 1])
    reward.to(device)
    r_max = -1e6

    r_multiplier = 1.5
    alpha = 0.5**(0.5)

    actor = lambda state: inference(
        torch.from_numpy(state).float().to(device), transition, reward, arglist.ensemble_size, 
        action_small_dim, arglist.plan_horizon, arglist.n_candidates, arglist.optimisation_iters, arglist.top_candidates, 
        device, r_max, r_multiplier, alpha
    )


    dataset = Dataset(arglist.ensemble_size, obs_dim_small, action_small_dim, arglist.batch_size_small, device, normalizer)


    trainer = Trainer(
        transition,
        reward,
        dataset,
        n_train_epochs=arglist.n_train_epoch_large,
        batch_size=arglist.batch_size_small,
        learning_rate=arglist.learning_rate,
        epsilon=arglist.epsilon,
        grad_clip_norm=arglist.grad_clip_norm,
    )
    # ------------------------------------------------------------- #





    print('step 2 The {} agents are inited ...'.format(1))

    print('step 3 starting iterations ...')

    # ------------------------------------------------------------- # 定义初始化
    var_cache_initial = 1
    var_small_initial = 1

    for episode_gone in range(2):
        env.episode = episode_gone
        obs_n_server_cache = env.reset_env()
        # print("obs_n_server_cache", obs_n_server_cache)
        action_n_server_cache = 0
        reward_server_n_cache = 0


        task_list1 = np.zeros(env.num_task)
        task_list2 = np.zeros(env.num_task)
        task_list3 = np.zeros(env.num_task)
        service_gain_list = np.zeros(env.num_task)

        for episode_cnt in range(arglist.per_episode_max_len):  # 每一回合里面的step 20*20数目

            #在这重制环境
            # game_step += 1
            print("this is the step " + str(episode_cnt) + " of episode " + str(episode_gone))
            var_cache_initial *= 0.9882
            # 500 步0.988
            current_cache = []
            for server in env.Server:
                current_cache.append(server.cache)
                # print("current_cache",current_cache)
            # caching decisionsM FORlstm-ddpg
            action_n_server_cache = twin_actor_agent_large.select_action(obs_n_server_cache)
            # print("action_n_server_cache",action_n_server_cache)
            # action_n_server_cache = agents_server_cache.select_action(obs_n_server_cache)
            # action_n_server_cache = action_n_server_cache[0].tolist()
            action_n_server_cache = np.clip(np.random.normal(action_n_server_cache, var_cache_initial), -1, 1)
            # print("noise: action_n_server_cache", action_n_server_cache)
            # print("NETWORK: action_n_server_cache", action_n_server_cache)
            # env._set_action_cache(action_n_server_cache)


            # print("swithc_cache_value",swithc_cache_value)
            constraint_over_cache_cap = 0
            for server in env.Server:
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.Tasks[i].get_cache_size
                #如果超出了 就不进行缓存更改
                if cache_size > server.get_cache_cap * 0.8:
                    # print("cache_size",cache_size/ (1024 * 1024 * 8))
                    # print("server.get_cache_cap",server.get_cache_cap/ (1024 * 1024 * 1024 * 8))
                    constraint_over_cache_cap += (cache_size - server.get_cache_cap_remain) / (1024 * 1024 * 1024 * 8)
                else:
                    constraint_over_cache_cap += 0
                    env._set_action_cache(action_n_server_cache)
                    server.cache_cap_remain = server.get_cache_cap - cache_size

            # 更新了大尺度的动作之后，就是可以计算处切换成本 
            swithc_cache_value = 0
            for i, server in enumerate(env.Server):
                cache = server.cache
                past_cache = current_cache[i]
                # print("past_cache",past_cache)
                # print("cache",cache)
                for i in range(len(cache)):
                    if cache[i] == 1 and past_cache[i] == 0:
                        # 因为我们设定cache_size = [0,50]
                        # print("task %d"%i,env.Tasks[i].get_cache_size/server.backhaul_rate)
                        swithc_cache_value += env.Tasks[i].get_cache_size / server.backhaul_rate

            # 进行小尺度的资源分配决策
            step_qoe = []
            step_time = []
            step_energy = []
            step_reward = []
            step_cost = []
            step_finish = []
            step_hit = []
            reward_mean = 0

            observation_space_samll = env.reset_env_small()
            print("----------------------------------")
            # print("observation_space_samll", observation_space_samll)
            for step in range(env.cache_decision_fre):

                for server in env.Server:
                    server.com_cap_remain = server.get_com_cap
                    server.bandwidth_cap_remain = server.get_bandwidth_cap

                # print("observation_space_samll", observation_space_samll)
                var_small_initial *= 0.9995
                # 这里做小尺度的卸载和通信和计算资源分配算法
                action_small = twin_actor_agent_small.select_action(observation_space_samll)
                #print("action_small", action_small.shape)
                action_small = np.clip(np.random.normal(action_small, var_small_initial), -1, 1)
                "这里需要一个将actionsmall" \
                "转化成off band fre 之类的东西，代替其遗传算法"
                #print(action_small)
                off, band, fre = env.trans_action_for_twin_ddpg(action_small, env.num_User, env.num_Server)
                # print("off, band, fre", off, band, fre)
                # sys.exit()
                env._set_action_offload(off)
                env._set_action_bandwidth(band)
                env._set_action_frequency(fre)

                # 计算奖励函数
                # 计算在这个动作下的时延和能耗
                total_time = 0
                total_energy = 0
                total_qoe = 0
                total_cost = 0
                total_cycle = 0
                total_band = 0
                total_finish = 0
                total_hit = 0
                total_gain = 0
                total_reward = 0
                # 开始计算奖励函数的所有用户的QOS 部分
                for i, user in enumerate(env.User):
                    # cloud time /and energy
                    # time_th = user.get_task_size / (100 * (1024 * 1024)) + user.get_task_cycle / (4 * 10 ** 9)


                    t_edge, e_edge, hit_num = env.ger_reward_time_energy(user, env)
                    total_hit += hit_num

                    # cost = 0.5 * (1 / t_edge) + 0.5 * (1 / e_edge)

                    cost = 1 / t_edge

                    if t_edge < env.Tasks[user.get_request - 1].get_tolerance_time:
                        total_finish += 1
                        # total_reward += cost
                    else:
                        total_finish += 0
                        # total_reward += 0
                    
                    total_reward += cost
                    # print("qoe", qoe)
                    total_cost += cost
                    # print("cost", cost)
                    # print("edge_qoe", qoe)
                    total_time += t_edge
                    total_energy += e_edge
                    # total_qoe += qoe
                    # 计算一下选择卸载的用户的缓存增益
                    service_gain_list[user.get_request - 1] += 1
                # 将这段动作不变时间内用户的服务质量求和
                # 因此需要加上这个时隙
                # 更新小时隙的环境状态
                #增加平衡项
                # print("total_cycle", total_cycle)

                # update the off popular
                off_state = []
                for user in env.User:
                    if user.offload ==1:
                        if env.Server[0].cache[user.request-1] ==1:
                            task_list1[user.request-1] += 1
                    if user.offload ==2:
                        if env.Server[1].cache[user.request-1] ==1:
                            task_list2[user.request-1] += 1
                    if user.offload ==3:
                        if env.Server[2].cache[user.request-1] ==1:
                            task_list3[user.request-1] += 1
                # print("!!!!!!!",task_list1)
                # print("**************",task_list2)
                for i in task_list1:
                    off_state.append(i)
                for i in task_list2:
                    off_state.append(i)
                for i in task_list3:
                    off_state.append(i)
                # normalization
                max_s = np.max(off_state)
                if np.min(off_state) == 0:
                    min_s = 0.1
                else:
                    min_s = np.min(off_state)
                for i in range(len(off_state)):
                    off_state[i] = (max_s - off_state[i]) / (max_s - min_s)
                env.service_offload_pop = off_state
                # update the caching_gain

                for i in range(len(service_gain_list)):
                    service_gain_list[i] = (service_gain_list[i] / env.Tasks[i].get_cache_size) * 10**10
                

                # 归一化这个服务缓存增益的变量
                max_s = np.max(service_gain_list)
                if np.min(service_gain_list) == 0:
                    min_s = 0.1
                else:
                    min_s = np.min(service_gain_list)
                for i in range(len(service_gain_list)):
                    service_gain_list[i] = (max_s - service_gain_list[i]) / (max_s - min_s)
                env.service_gain = service_gain_list
                ## update user request
                env.updat_user_requset()
                """

                ## 更新一下环境中的状态，缓存增益和卸载的决策
                ## env 更新一下用户的请求，然后再重新求解一下问题就好了
              """
                step_time.append(total_time)
                step_energy.append(total_energy)
                step_cost.append(total_cost)
                step_finish.append(total_finish)
                step_hit.append(total_hit)

                # print("cost swithc_cache_value constraint_over_cache_cap",np.mean(step_cost) / 10, swithc_cache_value / 1000, constraint_over_cache_cap / 100)
                # sys.exit()


                # if constraint_over_cache_cap > 0:
                #     reward_server_n_cache = 0
                # else:
                #     reward_server_n_cache = np.mean(
                #     step_cost) / 10 + np.mean(step_finish) - swithc_cache_value / 1000 - constraint_over_cache_cap / 100

                # print("finisha",total_finish)
                # total_gain = np.mean(total_finish / (cycle_gain + band_gain))

                # reward_samll = reward_samll * 10**3
                # reward_samll = 0.5 * total_reward + 0.5 * total_finish
                # reward_samll =  0.5 * (1 / total_time) +  0.5 * (1 / total_energy)
                reward_samll = 1 / total_time
                reward_samll = reward_samll * 10**4
                r_max = max(r_max, reward_samll)
                print("reward_samll", reward_samll)
                # reward_server_n_cache = total_qoe / 10 - swithc_cache_value / 1000 - constraint_over_cache_cap / 100
                # print("np.mean(step_qoe)/10",np.mean(step_qoe)/10)
                # print("swithc_cache_value/1000 ",swithc_cache_value/1000 )

                # print("the cache reward :", reward_server_n_cache)
                # next_obs_server_n_cache = env._get_obs()
                next_obs_server_n_cache_small = env._get_obs_samll()


                # print("next_obs_server_n_cache",next_obs_server_n_cache)
                # save experience

                # twin_actor_agent_small.replay_buffer.push((observation_space_samll, action_small,
                #                                      next_obs_server_n_cache_small, reward_samll))

                # buffer.add(observation_space_samll, action_small, reward_samll, next_obs_server_n_cache_small)
                dataset.add(observation_space_samll, action_small, next_obs_server_n_cache_small, reward_samll)


                observation_space_samll = next_obs_server_n_cache_small


            if constraint_over_cache_cap > 0:
                reward_server_n_cache = np.mean(step_hit) / 100  +  (1 / ( swithc_cache_value + 100) )
            else:
                reward_server_n_cache = np.mean(step_hit) / 100  +  (1 / ( swithc_cache_value + 100) ) 

            print("reward_server_n_cache", reward_server_n_cache)
            next_obs_server_n_cache = env._get_obs()
            twin_actor_agent_large.replay_buffer.push((obs_n_server_cache, action_n_server_cache,
                                                    next_obs_server_n_cache, reward_server_n_cache))

            obs_n_server_cache = next_obs_server_n_cache


    # esemble_loss, reward_loss = trainer.train(arglist.n_train_epochss)
    # twin_actor_agent_large.update(arglist)
    transition.reset()
    reward.reset()
    transition.to(device)
    reward.to(device)

    losses = []
    trans_losses = []
    rew_losses = []

    params = list(transition.parameters()) + list(reward.parameters())
    opt = torch.optim.Adam(
        params, lr=arglist.learning_rate, eps=arglist.epsilon
    )

    for epoch in range(arglist.n_train_epoch_large):
        
        for (states, actions, next_states, rewards) in dataset:

            transition.train()
            reward.train()

            transition_loss = transition.loss(states, 
                actions, next_states - states)
            reward_loss = reward.loss(states, actions, rewards)
            loss = transition_loss + reward_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, arglist.grad_clip_norm, norm_type=2
            )
            opt.step()

            losses.append(loss.item())
            trans_losses.append(transition_loss.item())
            rew_losses.append(reward_loss.item())

    print(f"Avg trans loss: {np.mean(trans_losses)}, avg rew loss: {np.mean(rew_losses)}")
    # t_loss, r_loss = trainer.train(arglist.n_train_epoch_large)
    print("初始化完成！")




    game_step = 0
    # 每一步用户和服务器的效用
    users_time = []
    users_energy = []
    users_qoe = []
    server_switch_cost = []
    rewards_server_cache = []
    rewards_samll = []
    users_cost = []
    users_cost1 = []
    users_finish = []
    users_hit = []
    ep_count_cache = 0

    t_losses = []
    r_losses = []

    # 刚开始的时候。是上下两层都进行重置！！！只用reset_high 既可
    # 获得到初始化的状态
    # max_epsode 500
    # each_episode: 200 step  做200 此的卸载和带宽分配决策d
    var_cache = 1
    var_small = 1
    for episode_gone in range(arglist.max_episode):
        env.episode = episode_gone
        obs_n_server_cache = env.reset_env()
        print("obs_n_server_cache", obs_n_server_cache)
        action_n_server_cache = 0
        reward_server_n_cache = 0
        ep_qoe = 0
        ep_cache_cost = 0
        ep_reward = 0
        ep_reward_small = 0
        ep_time = 0
        ep_energy = 0
        ep_cost = 0
        ep_cost1 = 0
        ep_finish = 0
        ep_hit = 0
        task_list1 = np.zeros(env.num_task)
        task_list2 = np.zeros(env.num_task)
        task_list3 = np.zeros(env.num_task)
        service_gain_list = np.zeros(env.num_task)



        for episode_cnt in range(arglist.per_episode_max_len):  # 每一回合里面的step 20*20数目

            #在这重制环境
            game_step += 1
            print("this is the step " + str(episode_cnt) + " of episode " + str(episode_gone))
            var_cache *= 0.9882
            # 500 步0.988
            current_cache = []
            for server in env.Server:
                current_cache.append(server.cache)
                # print("current_cache",current_cache)
            # caching decisionsM FORlstm-ddpg
            action_n_server_cache = twin_actor_agent_large.select_action(obs_n_server_cache)
            # action_n_server_cache = agents_server_cache.select_action(obs_n_server_cache)
            # action_n_server_cache = action_n_server_cache[0].tolist()
            action_n_server_cache = np.clip(np.random.normal(action_n_server_cache, var_cache), -1, 1)
            # print("noise: action_n_server_cache", action_n_server_cache)
            # print("NETWORK: action_n_server_cache", action_n_server_cache)
            # env._set_action_cache(action_n_server_cache)


            # print("swithc_cache_value",swithc_cache_value)
            constraint_over_cache_cap = 0
            for server in env.Server:
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.Tasks[i].get_cache_size
                #如果超出了 就不进行缓存更改
                if cache_size > server.get_cache_cap * 0.8:
                    # print("cache_size",cache_size/ (1024 * 1024 * 8))
                    # print("server.get_cache_cap",server.get_cache_cap/ (1024 * 1024 * 1024 * 8))
                    constraint_over_cache_cap += (cache_size - server.get_cache_cap_remain) / (1024 * 1024 * 1024 * 8)
                else:
                    constraint_over_cache_cap += 0
                    env._set_action_cache(action_n_server_cache)
                    server.cache_cap_remain = server.get_cache_cap - cache_size

            # 更新了大尺度的动作之后，就是可以计算处切换成本 
            swithc_cache_value = 0
            for i, server in enumerate(env.Server):
                cache = server.cache
                past_cache = current_cache[i]
                # print("past_cache",past_cache)
                # print("cache",cache)
                for i in range(len(cache)):
                    if cache[i] == 1 and past_cache[i] == 0:
                        # 因为我们设定cache_size = [0,50]
                        # print("task %d"%i,env.Tasks[i].get_cache_size/server.backhaul_rate)
                        swithc_cache_value += env.Tasks[i].get_cache_size / server.backhaul_rate

            # 进行小尺度的资源分配决策
            step_qoe = []
            step_time = []
            step_energy = []
            step_reward = []
            step_cost = []
            step_cost1 = []
            step_finish = []
            step_hit = []
            reward_mean = 0

            observation_space_samll = env.reset_env_small()
            print("----------------------------------")
            # print("observation_space_samll", observation_space_samll)
            for step in range(env.cache_decision_fre):
                # print("observation_space_samll", observation_space_samll)

                for server in env.Server:
                    server.com_cap_remain = server.get_com_cap
                    server.bandwidth_cap_remain = server.get_bandwidth_cap

                var_small *= 0.9995
                # 这里做小尺度的卸载和通信和计算资源分配算法
                with torch.no_grad():
                    action_small = actor(observation_space_samll)
                    action_small = action_small.cpu().detach().numpy()
                    #print("action_small", action_small.shape)
                action_small = np.clip(np.random.normal(action_small, var_small), -1, 1)
                "这里需要一个将actionsmall" \
                "转化成off band fre 之类的东西，代替其遗传算法"
                #print(action_small)
                off, band, fre = env.trans_action_for_twin_ddpg(action_small, env.num_User, env.num_Server)
                # print("off, band, fre", off, band, fre)
                # sys.exit()
                env._set_action_offload(off)
                env._set_action_bandwidth(band)
                env._set_action_frequency(fre)

                # 计算奖励函数
                # 计算在这个动作下的时延和能耗
                total_time = 0
                total_energy = 0
                total_qoe = 0
                total_cost = 0
                total_cost1 = 0
                total_cycle = 0
                total_band = 0
                total_finish = 0
                total_hit = 0
                total_gain = 0
                total_reward = 0
                # 开始计算奖励函数的所有用户的QOS 部分
                for i, user in enumerate(env.User):
                    # cloud time /and energy

                    t_edge, e_edge, hit_num = env.ger_reward_time_energy(user, env)
                    total_hit += hit_num

                    # cost = 1 / (0.5 * t_edge + e_edge * 0.5)
                    # print("t_edge e_edge", t_edge, e_edge)
                    cost1 = t_edge + e_edge
                    
                    # cost = 0.5 * (1 / t_edge) + 0.5 * (1 / e_edge)
                    cost = 1 / t_edge


                    if t_edge < env.Tasks[user.get_request - 1].get_tolerance_time:
                        total_finish += 1
                        # total_reward += cost
                    else:
                        total_finish += 0
                        # total_reward += 0

                    total_reward += cost
                    total_cost += cost
                    total_cost1 += cost1
                    # print("cost", cost)
                    # print("edge_qoe", qoe)
                    total_time += t_edge
                    total_energy += e_edge
                    # 计算一下选择卸载的用户的缓存增益
                    service_gain_list[user.get_request - 1] += 1
                # 将这段动作不变时间内用户的服务质量求和
                # 因此需要加上这个时隙
                # 更新小时隙的环境状态
                #增加平衡项
                # print("total_cycle", total_cycle)

                # update the off popular
                off_state = []
                for user in env.User:
                    if user.offload ==1:
                        if env.Server[0].cache[user.request-1] ==1:
                            task_list1[user.request-1] += 1
                    if user.offload ==2:
                        if env.Server[1].cache[user.request-1] ==1:
                            task_list2[user.request-1] += 1
                    if user.offload ==3:
                        if env.Server[2].cache[user.request-1] ==1:
                            task_list3[user.request-1] += 1
                # print("!!!!!!!",task_list1)
                # print("**************",task_list2)
                for i in task_list1:
                    off_state.append(i)
                for i in task_list2:
                    off_state.append(i)
                for i in task_list3:
                    off_state.append(i)
                # normalization
                max_s = np.max(off_state)
                if np.min(off_state) == 0:
                    min_s = 0.1
                else:
                    min_s = np.min(off_state)
                for i in range(len(off_state)):
                    off_state[i] = (max_s - off_state[i]) / (max_s - min_s)
                env.service_offload_pop = off_state
                # update the caching_gain

                for i in range(len(service_gain_list)):
                    service_gain_list[i] = (service_gain_list[i] / env.Tasks[i].get_cache_size) * 10**10
                

                # 归一化这个服务缓存增益的变量
                max_s = np.max(service_gain_list)
                if np.min(service_gain_list) == 0:
                    min_s = 0.1
                else:
                    min_s = np.min(service_gain_list)
                for i in range(len(service_gain_list)):
                    service_gain_list[i] = (max_s - service_gain_list[i]) / (max_s - min_s)
                env.service_gain = service_gain_list
                ## update user request
                env.updat_user_requset()
                """

                ## 更新一下环境中的状态，缓存增益和卸载的决策
                ## env 更新一下用户的请求，然后再重新求解一下问题就好了
              """
                # step_qoe.append(total_qoe)
                step_time.append(total_time)
                step_energy.append(total_energy)
                step_cost.append(total_cost)
                step_cost1.append(total_cost1)
                step_finish.append(total_finish)
                step_hit.append(total_hit)
                # print("cost swithc_cache_value constraint_over_cache_cap",np.mean(step_cost) / 10, swithc_cache_value / 1000, constraint_over_cache_cap / 100)
                # sys.exit()


                # if constraint_over_cache_cap > 0:
                #     reward_server_n_cache = 0
                # else:
                #     reward_server_n_cache = np.mean(
                #     step_cost) / 10 + np.mean(step_finish) - swithc_cache_value / 1000 - constraint_over_cache_cap / 100

                # print("finisha",total_finish)
                # total_gain = np.mean(total_finish / (cycle_gain + band_gain))

                # reward_samll = 0.5 * total_reward + 0.5 * total_finish
                reward_samll = 1 / total_time
                # reward_samll =  0.5 * (1 / total_time) +  0.5 * (1 / total_energy)
                reward_samll = reward_samll * 10**4
                print("reward_small", reward_samll)
                r_max = max(r_max, reward_samll)
                # print("r_max",r_max)
                # print("reward_samll, total_reward, total_finish", reward_samll, total_reward, total_finish)

                step_reward.append(reward_samll)


                # reward_server_n_cache = total_qoe / 10 - swithc_cache_value / 1000 - constraint_over_cache_cap / 100
                # print("np.mean(step_qoe)/10",np.mean(step_qoe)/10)
                # print("swithc_cache_value/1000 ",swithc_cache_value/1000 )

                # print("the cache reward :", reward_server_n_cache)
                # next_obs_server_n_cache = env._get_obs()

                next_obs_server_n_cache_small = env._get_obs_samll()

                # print("next_obs_server_n_cache",next_obs_server_n_cache)
                # save experience
                # twin_actor_agent_large.replay_buffer.push((obs_n_server_cache, action_n_server_cache,
                #                                      next_obs_server_n_cache, reward_server_n_cache))

                #主动推理经验池
                dataset.add(observation_space_samll, action_small, next_obs_server_n_cache_small, reward_samll)


                observation_space_samll = next_obs_server_n_cache_small

            

            if constraint_over_cache_cap > 0:
                reward_server_n_cache = np.mean(step_hit) / 100  +  (1 / ( swithc_cache_value + 100) )
            else:
                reward_server_n_cache = np.mean(step_hit) / 100  +  (1 / ( swithc_cache_value + 100) ) 


            print("reward_server_n_cache", reward_server_n_cache)
            next_obs_server_n_cache = env._get_obs()

            twin_actor_agent_large.replay_buffer.push((obs_n_server_cache, action_n_server_cache,
                                                    next_obs_server_n_cache, reward_server_n_cache))

            obs_n_server_cache = next_obs_server_n_cache


            if game_step >= arglist.learning_start_step:
                if game_step % arglist.learning_fre == 0:
                    print("learning !!!!!")
                    twin_actor_agent_large.update(arglist)

                    transition.reset()
                    reward.reset()
                    transition.to(device)
                    reward.to(device)

                    losses = []
                    trans_losses = []
                    rew_losses = []

                    params = list(transition.parameters()) + list(reward.parameters())
                    opt = torch.optim.Adam(
                        params, lr=arglist.learning_rate, eps=arglist.epsilon
                    )

                    for epoch in range(arglist.n_train_epoch_large):
                        
                        for (states, actions, next_states, rewards) in dataset:

                            transition.train()
                            reward.train()

                            transition_loss = transition.loss(states, 
                                actions, next_states - states)
                            reward_loss = reward.loss(states, actions, rewards)
                            loss = transition_loss + reward_loss
                            
                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                params, arglist.grad_clip_norm, norm_type=2
                            )
                            opt.step()

                            losses.append(loss.item())
                            trans_losses.append(transition_loss.item())
                            rew_losses.append(reward_loss.item())

                    print(f"Avg trans loss: {np.mean(trans_losses)}, avg rew loss: {np.mean(rew_losses)}")

            
            ep_cache_cost += swithc_cache_value
            ep_reward += reward_server_n_cache
            ep_reward_small += np.mean(step_reward)

            ep_time += np.mean(step_time)
            ep_energy += np.mean(step_energy)

            ep_cost += np.mean(step_cost)
            ep_cost1 += np.mean(step_cost1)
            # print("mean", np.mean(step_finish))
            ep_finish += np.mean(step_finish)
            ep_hit += np.mean(step_hit)
            



        print("ep_reward  ep_reward_small", ep_reward, ep_reward_small)
        # users_qoe.append(ep_qoe)
        rewards_server_cache.append(ep_reward)
        rewards_samll.append(ep_reward_small)
        server_switch_cost.append(ep_cache_cost)
        users_time.append(ep_time)
        users_energy.append(ep_energy)
        users_cost.append(ep_cost)
        users_cost1.append(ep_cost1)
        users_finish.append(ep_finish)
        users_hit.append(ep_hit)

        if episode_gone % 10 == 0:

            with open("simulation_edge/reward_large_user30_edge3_active2.txt", "w") as f:
                for r in rewards_server_cache:
                    f.write(str(r) + '\n')

            with open("simulation_edge/reward_small_user30_edge3_active2.txt", "w") as f:
                for r_small in rewards_samll:
                    f.write(str(r_small) + '\n')

            with open("simulation_edge/users_time_user30_edge3_active2.txt", "w") as f:
                for time in users_time:
                    f.write(str(time) + '\n')

            with open("simulation_edge/users_energy_user30_edge3_active2.txt", "w") as f:
                for energy in users_energy:
                    f.write(str(energy) + '\n')

            with open("simulation_edge/users_server_user30_edge3_active2.txt", "w") as f:
                for server in server_switch_cost:
                    f.write(str(server) + '\n')


            with open("simulation_edge/users_hit_user30_edge3_active2.txt", "w") as f:
                for hit in users_hit:
                    f.write(str(hit) + '\n')
                    

    return rewards_server_cache, server_switch_cost,rewards_samll, users_time, users_energy, users_cost, users_finish, users_hit





if __name__ == '__main__':
    arglist = parse_args()
    num_server = 2
    num_user = 20
    # agent1, agent2, agent_user,a1,a2,a6,a10,a17,a19,all_a = train(arglist)
    rewards_server_cache, server_switch_cost, rewards_samll, users_time, users_energy, users_cost, users_finish, users_hit = train(arglist)


    with open("simulation_edge/reward_large_user30_edge3_active2.txt", "w") as f:
        for r in rewards_server_cache:
            f.write(str(r) + '\n')

    with open("simulation_edge/reward_small_user30_edge3_active2.txt", "w") as f:
        for r_small in rewards_samll:
            f.write(str(r_small) + '\n')

    with open("simulation_edge/users_time_user30_edge3_active2.txt", "w") as f:
        for time in users_time:
            f.write(str(time) + '\n')

    with open("simulation_edge/users_energy_user30_edge3_active2.txt", "w") as f:
        for energy in users_energy:
            f.write(str(energy) + '\n')

    with open("simulation_edge/users_server_user30_edge3_active2.txt", "w") as f:
        for server in server_switch_cost:
            f.write(str(server) + '\n')


    with open("simulation_edge/users_hit_user30_edge3_active2.txt", "w") as f:
        for hit in users_hit:
            f.write(str(hit) + '\n')



    print('end')

