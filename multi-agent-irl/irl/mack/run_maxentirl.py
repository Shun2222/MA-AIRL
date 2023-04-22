#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym

import make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet

from .MaxEntIRL import *
from .libs import *
import pandas as pd
import json
import configparser
import pickle
from colorama import Fore, Back, Style

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    
    ACTION = "ACTION"
    MAIRL_PARAM = "MAIRL_PARAM"

    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    config_ini.read('./config/config.ini', encoding='utf-8')
    
    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation, nobs_flag=True)
    
    rewards = [[] for i in range(N_AGENTS)]
    irl = MaxEntIRL(env, N_AGENTS, config_ini)

    GAMMA = float(config_ini.get(MAIRL_PARAM, "GAMMA"))
    LEARNING_RATE = float(config_ini.get(MAIRL_PARAM, "LEARNING_RATE"))
    N_ITERS = int(config_ini.get(MAIRL_PARAM, "N_ITERS"))
    Seed_No = int(config_ini.get(MAIRL_PARAM, "Seed_No"))
    N_Seeds = int(config_ini.get(MAIRL_PARAM, "N_Seeds"))
    state = [str(i) for i in range(len(env[0].states))]
    save_dirs = []
    for count in range(N_Seeds):
        seed = Seed_No+count
        print("###### Now " + str(count/N_Seeds) + "% (Seed_No = "+ str(seed)+") ######")
        set_global_seeds(seed)
        feat_map = np.eye(irl.N_STATES)    
        logs = irl.maxent_irl(irl.N_STATES,irl.N_STATES,feat_map, experts, LEARNING_RATE, GAMMA, N_ITERS)
        save_dir = json.loads(config_ini.get("LOG", "SAVE_DIR"))
        save_dir = save(logs, seed, N_ITERS, STATE_SIZE, N_AGENTS, ENV, experts, save_dir)
        save_dirs.append(save_dir)
        identical=make_env.get_identical(env_id)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/atlas/u/lantaoyu/exps')
@click.option('--env', type=click.STRING, default='simple_spread')
@click.option('--expert_path', type=click.STRING,
              default='/atlas/u/lantaoyu/projects/MA-AIRL/mack/simple_spread/l-0.1-b-1000/seed-1/checkpoint20000-1000tra.pkl')
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']),
              default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--l2', type=click.FLOAT, default=0.1)
@click.option('--d_iters', type=click.INT, default=1)
@click.option('--rew_scale', type=click.FLOAT, default=0)
def main(logdir, env, expert_path, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, l2, d_iters,
         rew_scale):

    print(Fore.BLUE+"\n\
         __  __    _    ___ ____  _\n\
        |  \\/  |  / \\  |_ _|  _ \\| |\n\
        | |\\/| | / _ \\  | || |_) | |\n\
        | |  | |/ ___ \\ | ||  _ <| |___\n\
        |_|  |_/_/   \\_\\___|_| \\_\\_____|\n")

    print(Style.RESET_ALL)
    train()


if __name__ == "__main__":
    main()
