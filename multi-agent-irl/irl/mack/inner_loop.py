#Q学習
"""[Memo]
報酬関数が妥当でも，違った経路を先に学習してしまい，違う経路に収束してしまう
→MaxEntで違う経路の到達率は低くなるのでは？　→その経路も衝突はするが，解の一つで到達確率は減らない
→エキスパートでの違う経路の特徴が高すぎる
→違う経路を残すのと収束させるのはトレードオフ？（無限回学習させれば，正しい経路に収束するはず）
"""
from operator import is_
import pandas as pd
import numpy as np
import math
import copy
from tqdm import tqdm
import difflib,csv,datetime,os
from .libs.traj_util import *
from .archive import * 
from .libs.figure import *

class Q_learning():
    
    def __init__(self, env, N_AGENTS, config_ini):
        Q_PARAM = "Q_PARAM"
        self.step = np.zeros(N_AGENTS)
        self.step_in_multi = np.zeros(N_AGENTS)
        self.env = env # 引数で与えられたenvの代入
        self.RewardFunc = [[] for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            self.RewardFunc[i] = env[i].reward_func # envのrewardfuncに代入
        
        self.N_ROW = env.world.my_world_size[0].nrow #env row
        self.N_COL = env.world.my_world_size[1].ncol #env col
        self.N_AGENTS = N_AGENTS
        self.N_STATES = self.N_ROW * self.N_COL # state size
        self.N_ACTIONS = len(env[0]._actions) # 辞書式　"UP":0
        self.EPSILON     = float(config_ini.get(Q_PARAM, "EPSILON")) # epsilon-greedyのepsilon
        self.MAX_EPISODE = int(config_ini.get(Q_PARAM, "MAX_EPISODE")) # episode数
        self.LAST_EPISODE = self.MAX_EPISODE-2 #最後のエピソード番号
        self.LAMBDA      = float(config_ini.get(Q_PARAM, "LAMBDA")) #lambda 割引率
        self.ALPHA       = float(config_ini.get(Q_PARAM, "ALPHA")) # alpha
        self.MAX_STEP = self.N_STATES
        
        self.archive = Archive(N_AGENTS, config_ini);
        self.is_col_agents = []
        self.traj_gif = make_gif()
    
    # - ε-Greedy
    def actor(self, state, q_table, greedy=False):
        if not greedy:
            EPSILON = self.EPSILON
        else:
            EPSILON = -0.1

        if all((x == 0 for x in q_table[state])) or np.random.rand() < EPSILON: # 初めて訪れたとき＋ランダム
            act = np.random.choice(self.N_ACTIONS) # ランダムに行動
        else:
            act = np.argmax(q_table[state]) # 最大を選択
          
        return act
    
    
    #報酬を一回受け取ったかどうか　次の状態がすでに訪れたことがあるかで判定
    def bool_get_reward(self,state,past_history):
       if(state in past_history):
           return True
       else:
           return False

    def greedy_act_step(self, q_table, agents):
        self.step = np.zeros(self.N_AGENTS)
        self.step_in_multi = np.zeros(self.N_AGENTS)
        act = [[0] for i in range(self.N_AGENTS)]
        state = [self.env[i].start_pos for i in range(self.N_AGENTS)]
        next_state = [[0] for i in range(self.N_AGENTS)]
        end = [False for i in range(self.N_AGENTS)]
        history = [[] for i in range(self.N_AGENTS)]
        for i in range(self.N_AGENTS): # 各エージェントについて
            if agents[i].status != 'learning':
                continue
            for c in range(self.MAX_STEP):
                if end[i] == True: # goalしてたら何もしない
                    history[i].append(state[i]) # historyに現在の状態を追加
                    self.step[i] += 1
                    self.step_in_multi[i] += 1
                    break; 
                act[i] = self.actor(state[i],q_table[i],greedy=True)
                next_state[i] = self.env[i]._move(state[i],act[i]) # move
                end[i] = self.env[i].has_done(next_state[i]) # goalについたか？
                history[i].append(state[i]) # historyに現在の状態を追加
                self.step[i] += 1
                self.step_in_multi[i] += 1
                state[i] = next_state[i] # 現在状態を更新
        for i in range(self.N_AGENTS):
            if self.env[i].is_wall_traj(history[i]) or agents[i].status=='not_exist':
                history[i] = [-1]*self.MAX_STEP
                self.step[i] = len(history[i])
                self.step_in_multi[i] = len(history[i])
            if agents[i].status == 'learned':
                history[i] = copy.deepcopy(agents[i].greedy_act)
                self.step[i] = len(history[i])
                self.step_in_multi[i] = len(history[i])
        is_col = is_collision_matrix(history)
        for  i in range(self.N_AGENTS):
            if any(is_col[i]):
                self.step_in_multi[i] = self.MAX_STEP
        self.is_col_agents = is_col # 環境中にいるエージェントのみで判断すべき

        self.traj_gif.add_data(history)
        #print("greedy_state")
        #print(history)


    def run(self, experts=None, agents=None):
        act = [[0] for i in range(self.N_AGENTS)] # action
        state = [[0] for i in range(self.N_AGENTS)] # state
        next_state = [[0] for i in range(self.N_AGENTS)] # next state
        reward = [[0] for i in range(self.N_AGENTS)] # reward
        end = [False for i in range(self.N_AGENTS)] # end
        goalFlag = [False for i in range(self.N_AGENTS)] # goal flag
        history = [[] for i in range(self.N_AGENTS)] # previous state history
        self.expert = experts
        q_table = [[[0]*self.N_ACTIONS for i in range(self.N_STATES)] for j in range(self.N_AGENTS)]
        str_experts = [', '.join(map(str, experts[i])) for i in range(self.N_AGENTS)] # あるステップでの各エキスパート行動を,で連結

        """init"""
        for e in range(self.MAX_EPISODE): # エピソード回数実行
            for i in range(self.N_AGENTS): # 各エージェントについて初期化
                state[i] = self.env[i].start_pos
                end[i] = False
                goalFlag[i] = False
                history[i].clear()
            """training"""
            count = 0
            self.step = np.zeros(self.N_AGENTS)
            while (all(goalFlag)== False and count < self.MAX_STEP): # 全員がgoalするか、20stepで終了         
                """行動選択 次状態 報酬"""
                for i in range(self.N_AGENTS): # 各エージェントについて
                    if goalFlag[i] == True or agents[i].status != 'learning': # goalしてたら何もしない
                        continue 
                    act[i] = self.actor(state[i],q_table[i]) # actの獲得
                    next_state[i] = self.env[i]._move(state[i],act[i]) # move
                    end[i] = self.env[i].has_done(next_state[i]) # goalについたか？
                    reward[i] = self.env[i].get_reward(next_state[i]) # 即時報酬
                    history[i].append(state[i]) # historyに現在の状態を追加
                    self.step[i] += 1
                    if self.bool_get_reward(next_state[i], history[i]): # はじめて訪れるなら即時報酬は０
                        reward[i] = 0
                              
               
                """Q値の更新"""
                for i in range(self.N_AGENTS): # 各エージェントについて
                    if goalFlag[i] == True or agents[i].status != 'learning': # goalしてたら何もしない
                        continue
                    q_predict = q_table[i][state[i]][act[i]] # 将来の報酬期待値
                    if end[i] and not goalFlag[i]: # endであり、フィニッシュではない(ゴールして最後の更新)
                       q_target = reward[i] + self.LAMBDA * max(q_table[i][next_state[i]]) # gain なぜ＋１
                       q_table[i][state[i]][act[i]] += self.ALPHA * (q_target - (q_predict)) # 学習
                       state[i] = next_state[i] # 状態の更新
                       if not goalFlag[i]: # goalしていなかったら
                            history[i].append(next_state[i]) # historyに次状態を追加
                       goalFlag[i] = True # 最後の処理も終え、goalFlagを立てる
                    elif not end[i] and not goalFlag[i]: #まだゴールにたどり着いていないとき
                        if state[i] == next_state[i]: # 動いていなかったらマイナス報酬
                            #n_s = self.env[i].move_no_wall(state[i], act[i])
                            #if self.env[i].is_wall(n_s):
                            #    reward[i] = self.env[i].get_reward(n_s)
                            #else:
                            reward[i] = np.min(self.RewardFunc[i])
                        q_target = reward[i] + self.LAMBDA * max(q_table[i][next_state[i]]) # gain
                        q_table[i][state[i]][act[i]] += self.ALPHA * (q_target - (q_predict)) # 学習
                        state[i] = next_state[i] # 現在状態を更新
                count += 1

            """#訪れた最小のstep
            for i in range(self.N_AGENTS):
                if (self.step[i]==0 or self.step[i]>len(history[i])):
                    self.step[i] = len(history[i])
            """

            """アーカイブ"""
            #if all(goalFlag):
                #self.archive_traj.clear()
            for i in range(self.N_AGENTS):
                if self.env[i].is_wall_traj(history[i]) or agents[i].status=='not_exist':
                    history[i] = [-1]*self.MAX_STEP
                if agents[i].status == 'learned':
                    history[i] = copy.deepcopy(agents[i].greedy_act)
            self.archive.archive(history, experts)

        self.archive.count()
        self.greedy_act_step(q_table, agents)
        return q_table
    
    def q_learning(self,rewards, experts=None, agents=None):
        for i in range(self.N_AGENTS):
            self.env[i].reward_func = rewards[i] # 報酬の参照
            self.RewardFunc[i] = rewards[i] # 報酬の代入

        q_table = self.run(experts, agents) 
       
        
        return q_table