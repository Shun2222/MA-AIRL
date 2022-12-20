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
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
import seaborn as sns

@click.command()
@click.option('--env', type=click.STRING)

def plot_on_grid(values, file_name="Non", folder="./", set_annot=True, save=True, show=False, title=""):
    values = np.array(values)
    plt.figure()
    plt.title(title)
    img = sns.heatmap(values,annot=set_annot,square=True,cmap='PuRd')
    if save:
        file_path = make_path(folder, file_name)
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()
    return img

def create_graphs(env, path):
    file = "/atlas/u/lantaoyu/exps/airl/simple_path_finding/env6_3/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/"
    ite = f'{1400:05}'

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
    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs

    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0,
                               lr=0.25, max_grad_norm=0.5, kfac_clip=0.001,
                               lrschedule='linear', identical=None)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized' or disc_type == 'decentralized-all':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space,
                          state_only=True, discount=0.99, nstack=1, index=k, disc_type='decentralized',
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=5e7 // (1000 * 20),
                          lr_rate=0.001, l2_loss_ratio=l2) for k in range(num_agents)
        ]
    for i in range(num_agents):
        path = file + "d_"+str(i)+'_'+ite
        discriminator[i].load(path)
        obs = np.array([[-0.71244967, -1.81280087, -0.66666667, -0.75158051]])
        acs = np.array([[0,0,1,0,0]])
        rew = discriminator[i].ge_reward(obs, acs, None, None)
        print(rew)
        
fig.show()
if __name__ == '__main__':
    create_graphs()