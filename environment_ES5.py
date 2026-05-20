# 20240603
# by qian
# 建一个智能体学习的环境，然后这个是一个完全自主的开始，因为开始简化以前的描述，
# 使得整个方法只有一个类和一个环境和一个网络部分

import gym
from gym import spaces
import numpy as np
from core_1 import *

import sys


# environment for all agents in the multi-agent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self):
        self.episode = 0
        # how many steps, do one caching decision
        self.cache_decision_fre  = 5
        # how many steps, update the  task which user request
        self.task_update_fre = 1
        # 设置两个服务器智能体 20个用户智能体。固定 针对此环境 若改变智能体个数 则下面的环境reset需要进行改变
        # list of agents and entities (can change at execution-time!)
        self.num_User = 40
        self.num_Server =5
        self.num_task  = 10
        self.alpha = 0.5
        self.beta = 0.5
        self.xi = 1
        self.white_noise = -174  # dbm  #7月15日修改 原-114
        self.User = []
        #  this only one server
        self.Server = []
        self.Tasks = []
        # position dimensionality
        self.dim_p = 2
        self.tStoC = 0.02  # 100-200ms
        # large state for caching
        self.state = []
        self.state_small = []
        self.service_offload_pop  = None
        self.service_gain = None
        self.cache_state = None
        # add
        # edge server id from 1: location :two dimension ; com_cap: 20GHZ; pow : 1w
        # bandwidth_cap: 5MHZ; cache and next cache = [0];
        # cache_cap :300GB ;backhual_rate : 100Mbps（100mpbs 云端很远；）;
        # 在初始化定义时，所有的单位都是基本单位。bits\
        # def __init__(self, index, type, loc, power, com_cap, bandwidth_cap, cache, next_cache, cache_cap,
        #            backhaul_rate):
        self.Server = [Server((i + 1), 'server%d' % (i + 1), np.zeros(2), 1, 20 * (10 ** 9),
                               20 * (10 ** 7), np.zeros(self.num_task), np.zeros(self.num_task),
                               200 * (1024 * 1024 * 1024 * 8), 200 * (1024 * 1024 * 8)) for i in
                        range(self.num_Server)]
        # user id from 0; location :fixed two dimension; com_cap : 1GHZ; power:20 dbm ; energy consumption: 5*10**-27
        # def __init__(self, index, type, loc, power, compfre):
        self.User = [User(i, 'user %d' % i, np.zeros(2), 20,1 * (10 ** 9), 5 * (10 ** -27))
                      for i in range(self.num_User)]

        # def __init__(self, index, cache_size 800MB-4000MB , com_fre 400cycle/1000cycle, size, 50kB- 500KB)
        #__init__(self, index, cache_size, com_fre, size):
        self.Tasks = [Task((i + 1), 2000, 400, 500, 0.8) for i in range(self.num_task)]

        with open("task_characteristic.txt", "w") as f:
            total_size = 0
            total_cache_size = 0
            for task in self.Tasks:
                f.write("size" + str(task.get_size/ (1)) + "\t")
                f.write("cycle" + str(task.get_cycle/ (1)) + "\t")
                f.write("cache" + str(task.get_cache_size / ((1024 * 1024 *1024 * 8))) + "\n")
                total_cache_size += task.get_cache_size
                total_size += task.get_size
            f.write("total_cache_size" + str(total_cache_size/ (1024 * 1024 *1024 * 8)))
            f.write("total_size" + str(total_size/ (1024 * 1024 *1024 * 8)))

        # agent_action and state space
        self.action_space_server_cache = []
        self.observation_space_server_cache = []
        self.observation_space_samll = []



        self.ac_dim = self.num_task * self.num_Server
        self.action_space_server_cache.append(
            spaces.Box(low=-1, high=1, shape=(self.ac_dim,),
                       dtype=np.float32))

        self.obs_dim = self.num_task * (1+2*self.num_Server) + self.num_Server*1
        self.obs_dim_small = self.num_task * self.num_Server + self.num_Server*3 + self.num_User*3
        # cache state of number of server
        # question: why not define obs_dim directly
        self.observation_space_server_cache.append(
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))

        self.observation_space_samll.append(
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_small,), dtype=np.float32))
        # initialize the location of user and server
        # edge server is 5G BS ,the coverage diameter is 250 m



    # obtain the observation from the env
    def _get_obs(self):
        state = []
        cache_state = []
            
        for server in self.Server:
            for c in server.cache:
                cache_state.append(c)
        # set the initial state :[0]
        self.cache_state = cache_state
        for i in self.cache_state:
            state.append(i)
        for i in self.service_gain:
            state.append(i)
        for i in self.service_offload_pop:
            state.append(i)
                #加入服务器剩余缓存容量

        for server in self.Server:
            state.append(server.get_cache_cap_remain / 10**9)
            
        self.state = np.array(state)
        # print("len",len(self.state))
        return self.state

    def _get_obs_samll(self):
        state_small = []
        cache_state = []

        for server in self.Server:
            #这是把上一时刻缓存的cache放进缓存空间
            for c in server.cache:
                cache_state.append(c)


        # set the initial state :[0]
        self.cache_state = cache_state
        for i in self.cache_state:
            state_small.append(i)

        # #加入服务器剩余缓存容量
        # for server in self.Server:
        #     state.append(server.get_cache_cap_remain)
        for server in self.Server:
            state_small.append(server.get_cache_cap_remain / (1024*1024 *1024 * 8))
        #加入服务器剩余带宽
        for server in self.Server:
            state_small.append(server.get_bandwidth_cap_remain / 10**5)
        #加入服务器剩余计算资源
        for server in self.Server:
            state_small.append(server.get_com_cap_remain / 10**9)


        for user in self.User:
            state_small.append(user.get_request)

        for user in self.User:  
            state_small.append(user.get_task_size / 10**6)

        for user in self.User:
            state_small.append(user.get_task_cycle / 10**9)


        self.state_small = np.array(state_small)


        return self.state_small

    def distance(self):
        distance1 = np.zeros(self.num_User)
        distance2 = np.zeros(self.num_User)

        for j in range(self.num_User):
            distance1[j] = pow(pow((self.Server[0].get_loc[0] - self.User[j].get_loc[0]), 2) + pow((self.Server[0].get_loc[1] - self.User[j].get_loc[1]), 2), 0.5)

        for j in range(self.num_User):
            distance2[j] = pow(pow((self.Server[1].get_loc[0] - self.User[j].get_loc[0]), 2) + pow((self.Server[1].get_loc[1] - self.User[j].get_loc[1]), 2), 0.5)


        return distance1, distance2

    # initialize the environment
    def reset_env(self):
        print("reset_env!!!")
        np.random.seed(int(self.episode))
        #the communication range of the ES is 200*200 m
        # initialize the location of TDs
        for i in range(self.num_Server):
         self.Server[i].loc = [(2*i+1)*100,0]
        # initialize the location of TDs
        ave_user= self.num_User/self.num_Server
        for i in range(self.num_User):
            server_id  =  int(i/ave_user)
            loc_x = np.random.randint(low=-100, high=100) + self.Server[server_id].loc[0]
            loc_y = np.random.randint(low=-100, high=100)+ self.Server[server_id].loc[1]
            loc = []
            loc.append(loc_x)
            loc.append(loc_y)
            self.User[i].loc = loc
        # initialize the location of ES
        # clear the caching configration of ES
        cache = np.zeros(self.num_task)
        cache_state = []
        for server in self.Server:
            server.cache = cache
            server.next_cache = cache
            for c in server.cache:
                cache_state.append(c)
        # set the initial state :[0]
        self.cache_state = cache_state
        self.service_gain = np.zeros(self.num_task)
        self.service_offload_pop = np.zeros(self.num_task*self.num_Server)

        for server in self.Server:
            server.cache_cap_remain = server.cache_cap
            
        # reset the user's requset
        self.updat_user_requset()
        print("rest env end ")
        return self._get_obs()


    def reset_env_small(self):
        # print("reset_env_small!!!")
        # np.random.seed(int(self.episode))
        #the communication range of the ES is 200*200 m

        # initialize the location of TDs 初始化用户的设备的坐标
        ave_user= self.num_User/self.num_Server
        for i in range(self.num_User):
            server_id  =  int(i/ave_user)
            loc_x = np.random.randint(low=-100, high=100) + self.Server[server_id].loc[0]
            loc_y = np.random.randint(low=-100, high=100)+ self.Server[server_id].loc[1]
            loc = []
            loc.append(loc_x)
            loc.append(loc_y)
            self.User[i].loc = loc

        #初始化服务器的资源

        for server in self.Server:
            server.com_cap_remain = server.com_cap
            server.bandwidth_cap_remain = server.bandwidth_cap


        # self.Server[0].com_cap_remain = self.Server[0].com_cap
        # self.Server[0].bandwidth_cap_remain = self.Server[0].bandwidth_cap
 

        # self.Server[1].com_cap_remain = self.Server[1].com_cap
        # self.Server[1].bandwidth_cap_remain = self.Server[1].bandwidth_cap

        # self.Server[0].cache_cap_remain = self.Server[0].cache_cap
        # self.Server[1].cache_cap_remain = self.Server[1].cache_cap

        # reset the user's requset
        self.updat_user_requset()

        # print("rest env_small end ")
        return self._get_obs_samll()


    # Zipf distribution of user's request
    def number_of_certain_probability(self, sequence, probability):
        x = np.random.random()
        # print("x", x)
        cumulative_probability = 0.0
        # print("sequence", sequence)
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        # print(item)
        return item

    def updat_user_requset(self):
        for user in self.User:
            p = [0.21229198890532444, 0.1219297292172975, 0.08815289960395892, 0.07003023968857178,
                 0.05858111079665425,
                 0.050630545381643195, 0.04475640280685803, 0.04022181056515523, 0.036604931484490705,
                 0.033646012803007885,
                 0.031175932876210673, 0.029079612096248132, 0.02727588960294407, 0.02570580313991125,
                 0.02432543406547894,
                 0.023101363815598092, 0.02200769044526022, 0.02102401229050682, 0.02013403063536803,
                 0.019324559779512168]
            a = [self.num_task - i for i in range(self.num_task)]
            # print("a",a,p[0:self.num_task])
            user.request = self.number_of_certain_probability(a, p[0:self.num_task])
            user.task_size = self.Tasks[user.request - 1].get_size
            user.task_cycle = self.Tasks[user.request - 1].get_cycle

        ave_user= self.num_User/self.num_Server
        for i in range(self.num_User):
            server_id  =  int(i/ave_user)
            loc_x = np.random.randint(low=-100, high=100) + self.Server[server_id].loc[0]
            loc_y = np.random.randint(low=-100, high=100)+ self.Server[server_id].loc[1]
            loc = []
            loc.append(loc_x)
            loc.append(loc_y)
            self.User[i].loc = loc

    def _set_action_cache_popular(self,action):
        for i, server in enumerate(self.Server):
            server.cache =action
            print("after the maping :server%d.cache"%i,server.cache)
    def _set_action_cache_ga(self,action):
        cache = []
        # print("action", action)
        for i in action:
            if i > 0:
                cache.append(1)
            else:
                cache.append(0)
        for i, server in enumerate(self.Server):
            # print(cache[i * self.num_task:(i + 1) * self.num_task])
            server.cache = cache[i * self.num_task:(i + 1) * self.num_task]
            # print("after the maping :server%d.cache" % i, server.cache)
    def _set_action_cache(self, action):
        #print("action",action)
        cache = []

        for i in action:
            if i > 0:
                cache.append(1)
            else:
                cache.append(0)
        for i, server in enumerate(self.Server):
            # print(cache[i*self.num_task:(i+1)*self.num_task])
            server.cache = cache[i*self.num_task:(i+1)*self.num_task]
            # print("after the maping :server%d.cache"%i,server.cache)

    def _set_action_bandwidth(self,action):
        for i, band in enumerate(action):
            # 这里是[0,1] 的连续变量
            self.User[i].bandwidth = band

    def _set_action_frequency(self, action):
        for i, fre in enumerate(action):
            # 这里是[0,1] 的连续变量
            self.User[i].frequency = fre

    def _set_action_offload(self,action):
        # 这里是[0,1,2] 的离散的卸载
        for i, off in enumerate(action):
            self.User[i].offload = off



    # 这里只是一个简单的适用于2个智能体的
    def trans_action_for_twin_ddpg(self,action, users_num, server_num ):
        # print("small_action ", action)
        # action 是-1到1 之间的值
        o = action[0:users_num]
        o_new = np.zeros(users_num)

        # o_new = np.where(o < 0, 1, 2)  # [-1, 0) -> 服务器1, [0, 1] -> 服务器2

        o_new = np.where(o < -0.6, 1,
                np.where(o < -0.2, 2,
                np.where(o < 0.2, 3,
                np.where(o < 0.6, 4, 5))))

        # o < -1 -> 服务器1, -1 <= o < 0 -> 服务器2, 0 <= o < 1 -> 服务器3, o >= 1 -> 服务器4


        # print("o_new", o_new)

    # 提取带宽和频率，并确保非负且在 [0, 1] 范围内
        b = np.clip(action[users_num:2 * users_num], 0, 1)  # 限制在 [0, 1]
        f = np.clip(action[2 * users_num:3 * users_num], 0, 1)  # 限制在 [0, 1]


        user_alloc = [[] for _ in range(server_num)]
        for i in range(users_num):
            server_idx = o_new[i] - 1
            user_alloc[server_idx].append(i)

        for server_idx in range(server_num):
            users = user_alloc[server_idx]
            if not users:
                continue

            bw_total = sum(b[i] for i in users)
            freq_total = sum(f[i] for i in users)
            
            if bw_total == 0:
                bw_total = 1  # 防止除以0
            if freq_total == 0:
                freq_total = 1

            for i in users:
                b[i] = np.round(b[i] / bw_total, 3)
                f[i] = np.round(f[i] / freq_total, 3)


        return o_new, b, f


    # def trans_action_for_twin_ddpg(self, env, action, users_num, server_num, penalty_factor=0.001, min_ratio=1e-3, min_ratio1 = 1e-3):
    #     # if len(action) < 3 * users_num:
    #     #     raise ValueError("Action length must be at least 3 * users_num")
    #     # if server_num < 2:
    #     #     raise ValueError("Server number must be at least 2")


    #     o = action[:users_num]
    #     o_new = np.where(o < 0, 1, 2)  # 1代表服务器1，2代表服务器2

    #     # 初始资源请求

    #     b = np.clip(action[users_num:2 * users_num], 0, 1)
    #     f = np.clip(action[2 * users_num:], 0, 1)

    #     b = np.maximum(b, epsilon)
    #     f = np.maximum(f, epsilon)

    #     # 对 b 进行归一化
    #     b_sum = np.sum(b)
    #     if b_sum > 0:  # 防止除以 0
    #         b_normalized = b / b_sum
    #     else:
    #         b_normalized = b  # 如果总和为 0，保持原值（或可返回均匀分布）
        
    #     # 对 f 进行归一化
    #     f_sum = np.sum(f)
    #     if f_sum > 0:  # 防止除以 0
    #         f_normalized = f / f_sum
    #     else:
    #         f_normalized = f  # 如果总和为 0，保持原值（或可返回均匀分布）


    #     return o_new, b_normalized, , flag, band_gain, cycle_gain


    def trans_action_for_twin_ddpg1(self, env, action, users_num, server_num, penalty_factor=0.8, min_ratio=1e-2, min_ratio1 = 1e-2):
        # if len(action) < 3 * users_num:
        #     raise ValueError("Action length must be at least 3 * users_num")
        # if server_num < 2:
        #     raise ValueError("Server number must be at least 2")

        #记录是否超过剩余资源标志
        flag = 0
        flag_ = 0
        # 卸载决策：离散化
        o = action[:users_num]
        o_new = np.where(o < 0, 1, 2)  # 1代表服务器1，2代表服务器2

        # 初始资源请求
        b = np.clip(action[users_num:2 * users_num], 0, 1)
        b_ = action[users_num:2 * users_num]

        f1 = np.clip(action[2 * users_num:], 0, 1)

        # 初始化
        b_count = np.zeros(server_num)
        f_count = np.zeros(server_num)

        cycle_gain = np.zeros(server_num)
        band_gain = np.zeros(server_num)

        user_alloc = [[] for _ in range(server_num)]
        for i in range(users_num):
            server_idx = o_new[i] - 1
            user_alloc[server_idx].append(i)


        # print("user_alloc", user_alloc)
        for server_idx in range(server_num):
            users = user_alloc[server_idx]
            if not users:
                continue

            bw_remain = env.Server[server_idx].bandwidth_cap_remain
            freq_remain = env.Server[server_idx].com_cap_remain

            if bw_remain <= 0:
                print("出错", b, bw_remain)

            # print("bw_remain, freq_remain", bw_remain, freq_remain)

            #计算bw_total freq_total
            bw_total = sum(b[i] * bw_remain for i in users)
            freq_total = sum(f1[i] * freq_remain for i in users)

            # print("bw_total, freq_total", bw_total, freq_total)
            # --- 带宽修正 ---
            if bw_total > bw_remain:
                flag  = 1
                ratio_b = bw_remain / (bw_total + 1e-8)
                for i in users:
                    b[i] = b[i] * ratio_b * (1 - penalty_factor)  # 惩罚削弱
                    b[i] = max(b[i], min_ratio)
            else:
                for i in users:
                    b[i] = max(b[i], min_ratio)
            b_count[server_idx] = sum(b[i] * bw_remain for i in users)
            # print("b_count[server_idx]", b_count[server_idx])


            # --- 计算频率修正 ---
            if freq_total > freq_remain:
                flag = 1
                ratio = freq_remain / (freq_total + 1e-8)
                for i in users:
                    f1[i] = f1[i] * ratio * (1 - penalty_factor)
                    f1[i] = max(f1[i], min_ratio1)
            else:
                for i in users:
                    f1[i] = max(f1[i], min_ratio1)
            f_count[server_idx] = sum(f1[i] * freq_remain for i in users)
            # print("f_count[server_idx]", f_count[server_idx])


        for server_idx in range(server_num):
            cycle_gain[server_idx] = f_count[server_idx] / (env.Server[server_idx].get_com_cap)
            band_gain[server_idx] = b_count[server_idx] / (env.Server[server_idx].get_bandwidth_cap)



        for server_idx in range(server_num):
            bw_remain = env.Server[server_idx].bandwidth_cap_remain
            freq_remain = env.Server[server_idx].com_cap_remain
            users = user_alloc[server_idx]
            if not users:
                continue

            if b_count[server_idx] >= bw_remain:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                ratio_a = bw_remain / b_count[server_idx]
                for i in users:
                    b[i] = b[i] * ratio_a * 0.8

            if f_count[server_idx] >= freq_remain:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                ratio_b = freq_remain / f_count[server_idx]
                for i in users:
                    f1[i] = f1[i] * ratio_b * 0.8



        # 计算最终实际分配资源（以物理量表示）
        actual_bandwidth = np.zeros(users_num)
        actual_frequency = np.zeros(users_num)
        
        # if flag_ == 0:
        #     for i in range(users_num):
        #         server_idx = o_new[i] - 1
        #         actual_bandwidth[i] = b[i] * env.Server[server_idx].get_bandwidth_cap_remain
        #         actual_frequency[i] = f[i] * env.Server[server_idx].get_com_cap_remain
        # else:
        #     for i in range(users_num):
        #         server_idx = o_new[i] - 1
        #         actual_bandwidth[i] = 65536
        #         actual_frequency[i] = 983040

        # print("b f", b, f1)

        for i in range(users_num):
            server_idx = o_new[i] - 1
            actual_bandwidth[i] = b[i] * env.Server[server_idx].bandwidth_cap_remain
            actual_frequency[i] = f1[i] * env.Server[server_idx].com_cap_remain


            if b[i] <= 0 or env.Server[server_idx].bandwidth_cap_remain <= 0 or f1[i] <= 0 or env.Server[server_idx].com_cap_remain <= 0:
                x = env.Server[server_idx].bandwidth_cap_remain
                y = env.Server[server_idx].com_cap_remain
                log_content = (
                    f"daikuan: {b}, {x}, {y}, {b_}, {f1}\n"
                    "----------------------------------------\n"  # 分隔线
                )
                with open("simulation/error_log1.txt", "a") as f:  # 使用追加模式
                    f.write(log_content)
            

        # print("实际分配 剩余", b_count, f_count, env.Server[0].get_bandwidth_cap_remain, env.Server[0].get_com_cap_remain)


        return o_new, actual_bandwidth, actual_frequency, flag, band_gain, cycle_gain



    # def trans_action_for_twin_ddpg(self, action, users_num, server_num, min_freq_share=0.1):
    #     """
    #     将连续动作向量转换为卸载决策和资源分配比例，确保带宽和频率非负，无本地执行选项。

    #     参数:
    #         action: 长度为 3*users_num 的 numpy 数组，连续取值。
    #         users_num: 用户数。
    #         server_num: 服务端总数，目前支持 2。
    #         min_freq_share: 最小的频率分配比例，避免为 0。

    #     返回:
    #         o_new: 大小 users_num 的卸载决策数组，取值 1 或 2。
    #         b_new: 大小 users_num 的带宽分配比例数组。
    #         f_new: 大小 users_num 的频率分配比例数组。
    #     """
    #     # 提取动作并裁剪到 [0,1]
    #     o = action[:users_num]
    #     b = np.clip(action[users_num:2*users_num], 0.0, 1.0)
    #     f = np.clip(action[2*users_num:3*users_num], 0.0, 1.0)

    #     # 离散化卸载决策，只映射为 1 或 2
    #     # o < 0 -> 1, o >=0 -> 2
    #     o_new = np.where(o < 0, 1, 2)

    #     # 统计各服务器的带宽和频率强度
    #     b_count = {1: 0.0, 2: 0.0}
    #     f_count = {1: 0.0, 2: 0.0}
    #     for i in range(users_num):
    #         srv = o_new[i]
    #         b_count[srv] += b[i]
    #         f_count[srv] += f[i]

    #     # 避免分母为零并归一化总和最大为 1
    #     for srv in b_count:
    #         if b_count[srv] > 1.0:
    #             scale = 1.0 / b_count[srv]
    #             for i in range(users_num):
    #                 if o_new[i] == srv:
    #                     b[i] *= scale
    #             b_count[srv] = 1.0
    #         elif b_count[srv] == 0:
    #             b_count[srv] = 1.0

    #     for srv in f_count:
    #         if f_count[srv] > 1.0:
    #             scale = 1.0 / f_count[srv]
    #             for i in range(users_num):
    #                 if o_new[i] == srv:
    #                     f[i] *= scale
    #             f_count[srv] = 1.0
    #         elif f_count[srv] == 0:
    #             f_count[srv] = 1.0

    #     # 归一化分配比例
    #     b_new = np.zeros(users_num)
    #     f_new = np.zeros(users_num)
    #     for i in range(users_num):
    #         srv = o_new[i]
    #         b_share = b[i] / b_count[srv]
    #         b_new[i] = np.round(max(b_share, min_freq_share if f[i] > 0 else 0.0), 3)  # 保留三位小数
    #         # b_new[i] = np.round(b[i] / b_count[srv], 3)
    #         f_share = f[i] / f_count[srv]
    #         # 强制最小频率分配
    #         f_new[i] = np.round(max(f_share, min_freq_share if f[i] > 0 else 0.0), 3)

    #     return o_new, b_new.tolist(), f_new.tolist()




    def get_gain(self, server, user):
        ch_gains = 0
        server_loc = server.get_loc
        user_loc = user.get_loc
        # print('获取MBS与对应用户的增益', bs_loc, user_loc, bs_loc[0], user_loc[0], bs_loc[2], user_loc[1])
        distance = pow(pow((server_loc[0] - user_loc[0]), 2) + pow((server_loc[1] - user_loc[1]), 2), 0.5)
        if distance == 0:
            distance = 30
        ch_gains = 128.1 + 37.6 * np.log10(distance / 1000)  # m 转 distance KM
        # print("ch_gains:***************************ch_gains:")
        # print(ch_gains)
        # shadow = np.random.lognormal(0, 8, 1)
        # print(ch_gains, shadow)
        # shadow = 0
        ch_gains = pow(10, -ch_gains / 10)  # db  to w
        rayleigh = np.random.rayleigh(1)
        #rayleigh = 1
        gains = ch_gains * rayleigh  # w
        # print("gain",gains)
        return gains


    def ger_reward_time_energy(self,user, env):
        server_id = int(user.offload) - 1
        # print("user.offload",user.offload,"server_id",server_id)
        server = env.Server[server_id]

        num_server_user = int(env.num_User / env.num_Server)
        if user.get_id < num_server_user:
            server_associate = env.Server[0]
            # print("user_id", user.get_id, "server", env.server[0].get_type)
        elif user.get_id >= num_server_user and user.get_id < 2 * num_server_user:
            server_associate = env.Server[1]
            # print("user_id", user.get_id, "server", env.server[1].get_type)
        elif user.get_id >= 2 * num_server_user and user.get_id < 3 * num_server_user:
            server_associate = env.Server[2]
        elif user.get_id >= 3 * num_server_user and user.get_id < 4 * num_server_user:
            server_associate = env.Server[3]
        else:
            server_associate = env.Server[4]

        task_size = user.get_task_size
        task_cycle = user.get_task_cycle
        # print("task_size and task cycle",task_size,task_cycle)
        if user.get_frequency == 0:
            com_fre = server.get_com_cap * (user.get_frequency + 0.3)
        else:
            com_fre = server.get_com_cap * user.get_frequency
        # print("user frequence", user.get_frequency, )

        # edge computing time

        server.com_cap_remain = server.com_cap_remain - task_cycle

        if com_fre == 0:
            # print(user.get_type, user.get_offload)
            # print(user.get_frequency)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")
        t_exe = task_cycle / com_fre
        # print("t_exe",t_exe)
        # user uplink bandwidth
        if user.get_bandwidth == 0:
            bandwidth = server.get_bandwidth_cap * (user.get_bandwidth + 0.3)
        else:       
            bandwidth = user.get_bandwidth * server.get_bandwidth_cap

        server.bandwidth_cap_remain = server.bandwidth_cap_remain - task_size

        channel_gain = env.get_gain(server_associate, user)  # w
        power = pow(10, (user.get_power - 30) / 10)  # 20dbm to w - 0.1w
        self_gain = channel_gain * power
        white_noise_pow = pow(10, (-114 - 30) / 10)  # -144 dbm to w
        # uplink rate
        # print(np.log2(1 + self_gain / white_noise_pow))
        rate = bandwidth * np.log2(1 + self_gain / white_noise_pow)
        # from user to es  : transmission time
        if rate == 0:
            # print(user.get_type, user.get_offload)
            # print("userbandwidth", user.get_bandwidth)
            print("bandwidth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")

        # print("bandwidth and rate",user.bandwidth, rate)
        t_trans = task_size / rate

        hit_num = 0
        # print("transmission time", t_trans)

        # 这里是以瓦为单位还是以DBM 为单位呢？需要和本地计算的功率做一下比较
        e_trans = user.get_power * t_trans
        # print("e_trans", e_trans)
        if server.cache[user.get_request - 1] == 1:
            # print("cached all the service")
            t_total = t_exe + t_trans
            hit_num += 1
            # print("edge computing time", t_total)
        else:
            #  cloud computing frequency：4GHZ
            t_total = t_trans + task_size / (4 * 1024 * 1024) + task_cycle / (4 * 10 ** 9) + 0.1
            # print("cloud computing time", t_total)
        return t_total, e_trans, hit_num
