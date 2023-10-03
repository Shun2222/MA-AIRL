import gym
import click
import multiagent
import time
import tensorflow as tf
import make_env
import numpy as np
from rl.common.misc_util import set_global_seeds
from sandbox.mack.acktr_disc import Model, onehot
from sandbox.mack.policies import CategoricalPolicy
from rl import bench
import imageio
import pickle as pkl


@click.command()
@click.option('--env', type=click.STRING)
@click.option('--path', type=click.STRING, default="data/test.pkl")
@click.option('--discrete', is_flag=True)
@click.option('--grid_size', nargs=2, type=int, default=(0, 0))

def render(env, path, discrete, grid_size):
    tf.reset_default_graph()

    env_id = env

    def create_env():
        env = make_env.make_env(env_id, discrete_env=discrete, grid_size=grid_size)
        #env = make_env.make_env(env_id)
        env.seed(10)
        # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(10)
        return env

    env = create_env()
    #path = "/atlas/u/lantaoyu/exps//airl/simple_path_finding_single/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/m_01500" #2input single irl
    #path = r"/atlas/u/lantaoyu/exps/mack/simple_path_finding_single/l-0.1-b-1000/seed-1\checkpoint02400"
    #path = r"./data/checkpoint01500"
    #path = r"data/tag-dist-rew/checkpoint01100"
    #path = r"data/tag-dist-rew/airl/m_15000"
    trajs = pkl.load(open(path, 'rb'))

    n_agents = len(trajs[0]['ob'])
    ob_space = env.observation_space
    ac_space = env.action_space

    print('observation space')
    print(ob_space)
    print('action space')
    print(ac_space)

    n_actions = [action.n for action in ac_space]


    images = []
    num_trajs = len(trajs)

    for i in range(num_trajs):
        obs = env.reset()
        done = False
        for j in range(len(trajs[i]['ob'][0])):
            for k in range(n_agents):
                if all(trajs[i]['ob'][k][j] != obs[k]):
                    print('not match obs')

#             obs, rew, done, _ = env.step(actions_list)

            img = env.render_all_steps(mode='rgb_array')
            images.append(img[0])
            time.sleep(0.05)

    print(images.shape)
    imageio.mimsave(path + '.mp4', images, fps=25)


if __name__ == '__main__':
    render()
