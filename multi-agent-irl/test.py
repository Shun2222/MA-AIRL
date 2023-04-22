#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym
import make_env
from sandbox.mack.policies import CategoricalPolicy
from irl.mack.airl import *
from rl.acktr.utils import *
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from tqdm import tqdm
import seaborn as sns

@click.command()
@click.option('--env', type=click.STRING)
@click.option('--is_random', is_flag=True, default=False)
@click.option('--gif', is_flag=True, default=False)

def create_graphs(env, is_random, gif):
    #file = r"/atlas/u/lantaoyu/exps//airl/simple_path_finding_single/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/"
    #file = r"/atlas/u/lantaoyu/exps//airl/simple_path_finding/env6_3/2_obs/s-200/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/"
    #file = "\atlas\u\lantaoyu\exps\airl\simple_tag\decentralized\s-200\l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0\seed-1/"
    file = r"\atlas\u\lantaoyu\exps\airl\simple_tag\decentralized\s-10000\l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0\seed-1/"
    ite = f'{5000:05}'
    ite_range = [100, 10000]
    obs_label = ['pos_x', 'pos_y']

    tf.reset_default_graph()
    env_id = env
    def create_env():
        env = make_env.make_env(env_id)
        env.seed(10)
        # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(10)
        return env

    env = create_env()
   
    num_agents = len(env.action_space)
    #num_agents = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    num_acs = ac_space[0].n
    num_obs = 16
    if len(obs_label)!=num_obs:
        obs_label = []
        for i in range(num_obs):
            obs_label.append("obs"+str(i))

    make_model = lambda: Model(CategoricalPolicy, ob_space, ac_space, 1, 5e7, nprocs=1000, nsteps=20,
                               nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0,
                               lr=0.25, max_grad_norm=0.5, kfac_clip=0.001,
                               lrschedule='linear', identical=None)

    model = make_model()
    discriminator = [
        Discriminator(model.sess, ob_space, ac_space,
                      state_only=True, discount=0.99, nstack=1, index=k, disc_type='decentralized',
                      scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                      total_steps=5e7 // (1000 * 20),
                      lr_rate=0.001, l2_loss_ratio=0.1) for k in range(num_agents)
    ]

    obs = []
    acs = np.array([onehot(1, num_acs)])
    data = {}


    def base_n(num_10,n,max_num):
        str_n = ''
        while num_10:
            if num_10%n>=10:
                return -1
            str_n += str(num_10%n)
            num_10 //= n
        str_n = str_n[::-1].zfill(num_obs)
        res = []
        for s in str_n:
            res.append(int(s))
        res = np.array(res, dtype='float64')
        res -= (n-1)/2
        res = max_num * 2 * res/(n-1)
        return res

    for i in obs_label:
        data[i] = []
    n = 8
    if gif:
        print('fix later')
        """
        for i in range(num_agents):
            fig, ax = plt.subplots(figsize=(4, 4))
            plt.xlabel(obs_label[0]) # x軸ラベル
            plt.ylabel(obs_label[1]) # y軸ラベル
            frames = []
            rews = []
            print('making gif agent{}'.format(i))
            for iteration in range(ite_range[0], ite_range[1], 100):
                print('iteration{}'.format(iteration))
                path = file + "d_"+str(i)+'_'+f'{iteration:05}'
                discriminator[i].load(path)
                rew = discriminator[i].get_reward(obs, acs, None, None)
                x, y = np.meshgrid(values, values)
                im = ax.contourf(x, y, np.ravel(np.array(rew)).reshape(len(values), len(values))) # 塗りつぶし等高線図
                title = ax.text(0, 1.6, 'Iteration: {}'.format(iteration), fontsize=15) # 追加
                frames.append(im.collections+[title])
            anim = ArtistAnimation(fig, frames, interval=500)
            anim.save(file+"agent"+str(i)+".gif", writer="pillow")
            plt.close()"""
    else:
        for i in range(num_agents):
            print('agent{}'.format(i))
            path = file + "d_"+str(i)+'_'+ite
            discriminator[i].load(path)
            rew = []
            for j in obs_label:
                data[j] = []
            for o in tqdm(range(n**num_obs)):#n**num_obs)):
                obs = np.random.rand(1, num_obs) if is_random else np.array([base_n(o, n, 1.0)])
                pre_r = []
                for a in range(num_acs):
                    acs = np.array([onehot(1, num_acs)])
                    r = discriminator[i].get_reward(obs, acs, None, None)[0][0]
                    if a>0:
                        for r2 in pre_r:
                            if r!=r2:
                                print("Not same values")
                                print(f"agent{i}, action:{a}, obs:{obs}")
                    pre_r.append(r)

    return discriminator

if __name__ == '__main__':
    create_graphs()