import numpy as np
import pandas as pd
from itertools import product
from .inner_loop import Q_learning
from tqdm import tqdm
import csv,difflib,copy
import sys
import os
from pprint import pprint
import queue
from .agent import Agent
from .libs.figure import *
from .libs.traj_util import *

class MaxEntIRL():
     
    def __init__(self, env, N_AGENTS, config_ini):
        #環境
        MAXENT = "MAIRL_PARAM"
        self.env = [[] for i in range(N_AGENTS)]
        self.P_a = [[] for i in range(N_AGENTS)]
        for i in range(0,N_AGENTS):
            self.env[i] = env[i]
            self.P_a[i] = env[i].P        

        self.N_AGENTS = N_AGENTS 
        self.N_ROW = env[0].nrow
        self.N_COL = env[0].ncol
        self.N_STATES = self.N_ROW * self.N_COL
        self.N_ACTIONS = len(env[0]._actions)  
        
        self.config_ini = config_ini
        self.reward_func = None 
        self.agents = None
        self.inner_loop = None
    
    """0~1に変換"""
    def normalize(self,vals):
        min_val = np.min(vals)
        max_val = np.max(vals)
        if max_val-min_val == 0:
            cv_vals = vals - min_val
        else:
            cv_vals = (vals - min_val) / (max_val - min_val)
        return np.round(cv_vals,4)

    
    #def trans_mat(self, env):
        # P[state][action] = [(prob, next_state, reward)] P[s][a][0][1] = next_state
        # n_states * n_actions * n_states next_stateにビットがったったもの one-hot_state
    #    return np.array([[np.eye(1, self.N_STATES, env.P[s][a][0][1])[0] if env.P[s][a] else np.zeros(self.N_STATES) for a in range(self.N_ACTIONS)] for s in range(self.N_STATES)])
        
    """行動系列の状態到達回数"""
    # 行動を全て同じ報酬で与える　ここはゴールに近いほど高いほうが自由度上がりそう　それは報酬推定でやることでここではない 
    def calculate_state_visition_count(self, trajs):
        features = np.zeros(self.N_STATES)
        for t in trajs:
            for s in t:
                features[s] += 1
        return features
    
    def compute_policy(self, Qtable):
        policy = np.zeros([self.N_STATES, self.N_ACTIONS])
        for s in range(self.N_STATES):
            v_s = np.array(Qtable[s]) 
            if all(x == 0 for x in v_s): # 初めて到達したとき
                policy[s,:] = np.transpose(v_s) # agent_iの状態sについての中身全部
            else: #2回目以降はその行動を取った確率
                 policy[s,:] = np.transpose(v_s/np.sum(v_s))
        return policy

    """パラメータθの基での状態到達頻度"""
    def compute_state_visition_freq(self,trans_probs, trajs, policy):       
        #trajs = [traj0, traj1, ...], traj = [s0, s1, ...]
        n_t = len(trajs[0]) # 0番目行動系列の大きさ(step) 
        mu = np.zeros((self.N_STATES, n_t)) # 状態数*step
        for traj in trajs:
            mu[traj[0], 0] += 1 # 0step目の状態を足していく
        mu[:, 0] = mu[:, 0] / len(trajs) # 正規化
        for t in range(1, n_t):
            for pre_s, a, s in product(range(self.N_STATES), range(self.N_ACTIONS), range(self.N_STATES)):
                # ある状態での次ステップに到達する確率 = ひとつ前の状態とステップの占有率*状態遷移map*ある行動をとる確率　の合計
                mu[s, t] += (mu[pre_s, t-1]* trans_probs[pre_s, a, s]*policy[pre_s, a])
        return np.sum(mu, axis=1) #各状態でのstepに到達する確率の合計
    
    """ sum collision_num/compair_num * traj """
    """ただの行動と組量ではなく、状態到達頻度確率であることに注意 ただの入れ替えではなく、update方式も検討"""

    def merge_feature(self, agent_num):
        count_memory = self.inner_loop.archive.count_memory
        opt_traj_archive = self.inner_loop.archive.opt_traj_archive

        if not opt_traj_archive[agent_num]:
            feature = self.calculate_state_visition_count(self.agents[agent_num].original_expert)/len(self.agents[agent_num].original_expert)
            return feature

        max_non_col = None
        traj = None
        for t in opt_traj_archive[agent_num]:
            str_traj = array_to_str(t)
            sum_col = 0
            sum_non_col = 0
            col_rate = 1.0
            for i in range(self.N_AGENTS):
                if count_memory[agent_num][i][str_traj]:
                    sum_col = count_memory[agent_num][i][str_traj][0]
                    sum_non_col = count_memory[agent_num][i][str_traj][1]
                    col_rate *= sum_non_col/(sum_col+sum_non_col) if sum_col+sum_non_col!=0 else 1.0
            if not max_non_col:
                max_non_col = col_rate
                traj = str_to_array(str_traj)
            else:
                if max_non_col < col_rate:
                    max_non_col = col_rate
                    traj = str_to_array(str_traj)

        feature = self.calculate_state_visition_count([traj])
        return feature

    def merge_feature_by_rank(self, agent_num):
        count_memory = self.inner_loop.archive.count_memory
        opt_traj_archive = self.inner_loop.archive.opt_traj_archive

        rank = self.create_rank()
        if agent_num==rank[0] or not opt_traj_archive[agent_num]:
            feature = self.calculate_state_visition_count(self.agents[agent_num].original_expert)/len(self.agents[agent_num].original_expert)
            return feature


        i = 0
        lower_rank = []
        while True:
            if agent_num==rank[i]:
                break
            lower_rank.append(rank[i])
            i += 1

        max_non_col = None
        traj = None
        for t in opt_traj_archive[agent_num]:
            str_traj = array_to_str(t)
            col = 0
            non_col = 0
            non_col_rate = 1.0
            for i in lower_rank:
                if count_memory[agent_num][i][str_traj]:
                    col = count_memory[agent_num][i][str_traj][0]
                    non_col = count_memory[agent_num][i][str_traj][1]
                    non_col_rate *= non_col/(col+non_col) if col+non_col!=0 else 0.0
            if not max_non_col:
                max_non_col = non_col_rate
                traj = str_to_array(str_traj)
            else:
                if max_non_col < non_col_rate:
                    max_non_col = non_col_rate
                    traj = str_to_array(str_traj)

        feature = self.calculate_state_visition_count([traj])
        return feature

    def update_expert(self, use_rank=True, update_rate=None):
        if use_rank:
            for i in range(self.N_AGENTS):
                feature = self.merge_feature_by_rank(i)
        else:
            for i in range(self.N_AGENTS):
                feature = self.merge_feature(i) 
                
        if update_rate!=None:
            self.agents[i].feature_expert = (1-update_rate)*self.agents[i].feature_expert + update_rate*feature
            self.agents[i].feature_expert = self.normalize(self.agents[i].feature_expert)
        else:
            self.agents[i].feature_expert = feature

    def freedom(self):
        opt_traj_archive = self.inner_loop.archive.opt_traj_archive
        f = [len(opt_traj_archive[i]) for i in range(self.N_AGENTS)]
        return np.array(f)

    def create_rank(self):
        freedom = self.freedom()
        priority_rank = [-1 for _ in range(self.N_AGENTS)]
        i = 0
        while True:
            if i==self.N_AGENTS: 
                break
            r = np.where(freedom==np.sort(freedom)[i])[0]
            for j in r:    
               if not j.item() in priority_rank:
                priority_rank[i] = j.item()
                i += 1
        return priority_rank

    def maxent_irl(self, N_STATES, N_ACTIONS, feat_map, experts, lr, GAMMA, n_iters):

        self.reward_func = [np.zeros(self.N_STATES) for i in range(self.N_AGENTS)]
        self.agents = [Agent(i) for i in range(self.N_AGENTS)]
        self.inner_loop = Q_learning(self.env,self.N_AGENTS, self.config_ini)

        # init parameters
        Qtables = {}
        theta = [np.zeros((feat_map.shape[1],)) for _ in range(self.N_AGENTS)]
        trans_probs = [[] for _ in range(self.N_AGENTS)]
        pre_iters = 20
        sum_iters = pre_iters+n_iters
        step_hist = [[] for _ in range(self.N_AGENTS)]
        step_in_multi_hist = [[] for _ in range(self.N_AGENTS)]
        expert_gifs = [make_gif() for _ in range(self.N_AGENTS)]
        agent_memory = []
        col_count = [np.zeros(self.N_AGENTS) for _ in range(self.N_AGENTS)]
        col_greedy = [[] for _ in range(self.N_AGENTS)]
        rank_hist = []


        for i in range(self.N_AGENTS):
            trans_probs[i] = self.trans_mat(self.env[i]) # 状態遷移map
            self.agents[i].original_expert = experts[i]
            self.agents[i].feature_expert = self.calculate_state_visition_count(experts[i])/len(experts[i])
            self.agents[i].status = 'learning'

        print("Start learning")
        for iteration in tqdm(range(int(n_iters))): 
            """inner loop"""
            Qtables = self.inner_loop.q_learning(experts=experts, rewards=self.reward_func, agents=self.agents)       
            """learn"""
            #Q値→方策→状態到達頻度確率→エキスパートとの差→報酬修正
            for i in range(self.N_AGENTS): 
                Qtable = self.normalize(Qtables[i]) 
                policy = self.compute_policy(Qtable)
                # compute state visition requences 各状態にあるステップで到達する確率の合計
                svf = self.compute_state_visition_freq(trans_probs[i], experts[i], policy) 
                p_svf = feat_map.T.dot(svf)  

                self.update_expert(); # エキスパート行動の生成
                grad = self.agents[i].feature_expert - p_svf
                theta[i] += lr * grad
                theta[i] = np.round(theta[i], 4)
                self.reward_func[i] = feat_map.dot(theta[i].T)

            for i in range(self.N_AGENTS):
                step_hist[i].append(copy.deepcopy(self.inner_loop.step[i]))
                step_in_multi_hist[i].append(copy.deepcopy(self.inner_loop.step_in_multi[i]))
                expert_gifs[i].add_data(copy.deepcopy(self.agents[i].feature_expert))
                sum_col = 0
                for j in range(self.N_AGENTS):
                    if self.inner_loop.is_col_agents[i][j]:
                        col_count[i][j] += 1
                        sum_col += 1
                col_greedy[i].append(sum_col)
            agent_memory.append(copy.deepcopy(self.inner_loop.archive.count_memory))
            rank_hist.append(self.create_rank())
    
            #print("Memory")
            #self.inner_loop.archive.print_count_memory()
            #self.inner_loop.archive.clear_memory()

        logs = {
            "rewards" : self.reward_func,
            "feat_experts" : [self.agents[i].feature_expert for i in range(self.N_AGENTS)],
            "step_hist" : step_hist,
            "step_in_multi_hist" : step_in_multi_hist,
            "expert_gifs" : expert_gifs,
            "agent_memory" : agent_memory,
            "col_count" : col_count,
            "col_greedy" : col_greedy,
            "traj_gif" : self.inner_loop.traj_gif, 
            "rank" : rank_hist, 
            "agetns" : self.agents
            }
        
        return 