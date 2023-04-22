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
from sandbox.mack.acktr_disc import learn
from sandbox.mack.policies import CategoricalPolicy
from irl.mack.kfac_discriminator_airl import Discriminator
import irl.mack.airl as airl
from rl.acktr.utils import *
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, discriminator_dir, irl_iteration):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            discriminator = load_discriminator(env, discriminator_dir, irl_iteration)
            env = make_env.make_env(env_id, discriminator=discriminator)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    
    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = CategoricalPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=make_env.get_identical(env_id))
    env.close()

def load_discriminator(env, load_dir, iteration):
    ite = f'{iteration:05}'
    tf.reset_default_graph()
   
    num_agents = len(env.action_space)
    #num_agents = 1
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda: airl.Model(CategoricalPolicy, ob_space, ac_space, 1, 5e7, nprocs=1000, nsteps=20,
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


    for i in range(num_agents):
        path = load_dir + "d_"+str(i)+'_'+ite
        discriminator[i].load(path)

    return discriminator

@click.command()
@click.option('--logdir', type=click.STRING, default='/irl/test')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener',
                                          'simple_crypto', 'simple_push',
                                          'simple_tag', 'simple_spread', 'simple_adversary',
                                           'simple_path_finding', 'simple_path_finding_single',
                                           'simple_test']))
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--iteration', type=click.INT, default=10000)#5e7
@click.option('--rewdir', type=click.STRING, default=r"\atlas\u\lantaoyu\exps\airl\simple_tag\decentralized\s-1000\l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0\seed-1/")
@click.option('--irl_iteration', type=click.INT, default=1000)
def main(logdir, env, lr, seed, batch_size, atlas, iteration, rewdir, irl_iteration):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]
    total_timesteps = iteration*1000 #default:5e7
    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/exps/mack/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
              env_id, total_timesteps, lr, batch_size, seed, batch_size // 250, rewdir, irl_iteration)


if __name__ == "__main__":
    main()
