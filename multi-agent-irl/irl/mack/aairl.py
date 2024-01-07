import os.path as osp
import random
import time
import pickle as pkl
from decimal import *

import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from rl.acktr.utils import Scheduler, find_trainable_variables, discount_with_dones
from rl.acktr.utils import cat_entropy, mse, onehot, multionehot

from rl import logger
from rl.acktr import kfac
from rl.common import set_global_seeds, explained_variance
from irl.mack.kfac_discriminator_airl import Discriminator
from irl.dataset import Dset
from gym import spaces

ARC_INDI_THRESHOLD = 250
GOAL_STEP_THRESHOLD = 15 
MAX_ARC = 3

def sort_arc(arc, scores):
    idxs = sorted(range(len(scores)), key=lambda i: scores[i])
    sorted_arc = [arc[i] for i in idxs]
    sorted_scores = [scores[i] for i in idxs]
    return sorted_arc, sorted_scores
    
def push_arc(arc, target, arc_scores, target_score):

    if np.min(arc_scores)>target_score:
        return arc, arc_scores
    
    arc[0] = target
    arc_scores[0] = target_score
    return sort_arc(arc, arc_scores)

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nbatch = nenvs * nsteps
        self.num_agents = num_agents = len(ob_space)
        self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        if identical is None:
            identical = [False for _ in range(self.num_agents)]

        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents

        A, ADV, R, PG_LR = [], [], [], []
        for k in range(num_agents):
            if identical[k]:
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
            else:
                A.append(tf.placeholder(tf.int32, [nbatch * scale[k]]))
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, []))

        pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = []
        self.model2 = train_model = []
        self.pg_fisher = pg_fisher_loss = []
        self.logits = logits = []
        self.sample_net = sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = []
        self.lld = lld = []
        self.log_pac = []

        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1])
            else:
                step_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                         nenvs, 1, nstack, reuse=False, name='%d' % k))
                train_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                          nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))
            logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_model[k].pi, labels=A[k])
            self.log_pac.append(-logpac)

            lld.append(tf.reduce_mean(logpac))
            logits.append(train_model[k].pi)

            pg_loss.append(tf.reduce_mean(ADV[k] * logpac))
            entropy.append(tf.reduce_mean(cat_entropy(train_model[k].pi)))
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k]
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k])))
            train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])

            pg_fisher_loss.append(-tf.reduce_mean(logpac))
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf)))
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2)))
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k])

        self.policy_params = []
        self.value_params = []

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))
        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]
        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ]

        self.optim = optim = []
        self.clones = clones = []
        update_stats_op = []
        train_op, clone_op, q_runner = [], [], []

        for k in range(num_agents):
            if identical[k]:
                optim.append(optim[-1])
                train_op.append(train_op[-1])
                q_runner.append(q_runner[-1])
                clones.append(clones[-1])
                clone_op.append(clone_op[-1])
            else:
                with tf.variable_scope('optim_%d' % k):
                    optim.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(optim[k].compute_and_apply_stats(joint_fisher_loss, var_list=params[k]))
                    train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                    train_op.append(train_op_)
                    q_runner.append(q_runner_)

                with tf.variable_scope('clone_%d' % k):
                    clones.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(clones[k].compute_and_apply_stats(
                        pg_fisher_loss[k], var_list=self.policy_params[k]))
                    clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    clone_op.append(clone_op_)

        update_stats_op = tf.group(*update_stats_op)
        train_ops = train_op
        clone_ops = clone_op
        train_op = tf.group(*train_op)
        clone_op = tf.group(*clone_op)

        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = [rewards[k] - values[k] for k in range(num_agents)]
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1)

            td_map = {}
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = []
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([multionehot(actions[i], self.n_actions[i])
                                                   for i in range(num_agents) if i != k], axis=1))
                    action_v = np.concatenate(action_v, axis=0)
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})

                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].X_v: np.concatenate([ob.copy() for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    ADV[k]: np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0),
                    R[k]: np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                try:
                    sess.run(train_ops[k], feed_dict=new_map)
                except:
                    print("Failed to train generator!!!")
                td_map.update(new_map)

                if states[k] != []:
                    td_map[train_model[k].S] = states
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy = sess.run(
                [pg_loss, vf_loss, entropy],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            td_map = {}
            cur_lr = self.clone_lr.value()
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(clone_ops[k], feed_dict=new_map)
                td_map.update(new_map)
            lld_loss = sess.run([lld], td_map)
            return lld_loss

        def get_log_action_prob(obs, actions):
            action_prob = []
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)
                }
                log_pac = sess.run(self.log_pac[k], feed_dict=new_map)
                if scale[k] == 1:
                    action_prob.append(log_pac)
                else:
                    log_pac = np.split(log_pac, scale[k], axis=0)
                    action_prob += log_pac
            return action_prob

        self.get_log_action_prob = get_log_action_prob

        def get_log_action_prob_step(obs, actions):
            action_prob = []
            for k in range(num_agents):
                action_prob.append(step_model[k].step_log_prob(obs[k], actions[k]))
            return action_prob

        self.get_log_action_prob_step = get_log_action_prob_step

        def save(save_path):
            ps = sess.run(params_flat)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params_flat, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob, av, *_args, **_kwargs):
            a, v, s = [], [], []
            obs = np.concatenate(ob, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                      for i in range(num_agents) if i != k], axis=1)
                a_, v_, s_ = step_model[k].step(ob[k], obs, a_v)
                a.append(a_)
                v.append(v_)
                s.append(s_)
            return a, v, s

        self.step = step

        def value(obs, av):
            v = []
            ob = np.concatenate(obs, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                      for i in range(num_agents) if i != k], axis=1)
                v_ = step_model[k].value(ob, a_v)
                v.append(v_)
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type, nobs_flag=False):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.nobs_flag = nobs_flag
        self.num_agents = len(env.observation_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((nenv, )) for _ in range(self.num_agents)]
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        #ac_space = env.action_space
        ac_space = [spaces.Discrete(4) for _ in range(2)]
        self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        self.dones = [np.array([False for _ in range(nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs
        # for k in range(self.num_agents):
        #     ob = np.roll(self.obs[k], shift=-1, axis=1)
        #     ob[:, -1] = obs[:, 0]
        #     self.obs[k] = ob

        # self.obs = [np.roll(ob, shift=-1, axis=3) for ob in self.obs]
        # self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_obs_next = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_report_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_is_goal = [[] for _ in range(self.num_agents)]
        mb_is_collision = [[] for _ in range(self.num_agents)]
        mb_states = self.states

        arc_indi_obs = [[] for _ in range(self.num_agents)]
        arc_indi_obs_next = [[] for _ in range(self.num_agents)]
        arc_indi_all_obs = [[] for _ in range(self.num_agents)]
        arc_indi_actions = [[] for _ in range(self.num_agents)]
        arc_indi_values = [[] for _ in range(self.num_agents)]
        arc_indi_scores = [[] for _ in range(self.num_agents)]
        arc_coop_obs = [[] for _ in range(self.num_agents)]
        arc_coop_obs_next = [[] for _ in range(self.num_agents)]
        arc_coop_all_obs = [[] for _ in range(self.num_agents)]
        arc_coop_actions = [[] for _ in range(self.num_agents)]
        arc_coop_values = [[] for _ in range(self.num_agents)]
        arc_coop_scores = [[] for _ in range(self.num_agents)]

        obs = self.env.reset()
        self.dones = [np.array([False for _ in range(self.nenv)]) for k in range(self.num_agents)]
        self.update_obs(obs)
        self.actions = [np.zeros((self.nenv, )) for _ in range(self.num_agents)]

        traj_index = 0
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)
            self.actions = actions

            not_all_done = False 
            for k in range(self.num_agents):
                if not self.dones[0][k]:
                    not_all_done = True
            if not not_all_done:
                # 初期化されると観測値に0.0が帰ってくるのでここで捨てる
                traj_index += 1
                obs = self.env.reset()
                self.update_obs(obs)
                #self.info = [{} for _ in range(self.nenv)]

            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])



            actions_list = []
            for i in range(self.nenv):
                actions_list.append([onehot(actions[k][i], self.n_actions[k]) for k in range(self.num_agents)])

            obs, true_rewards, dones, info = self.env.step(actions_list)

            is_round = True
            for k in range(self.num_agents):
                for l in range(len(obs[k])):
                    if is_round:
                        obs[k][l] = np.array([round(Decimal(obs[k][l][0]), 2), round(Decimal(obs[k][l][1]), 2)])
                for ni, done in enumerate(dones[k]):
                    if done:
                        obs[k][ni] = obs[k][ni] * 0.0
            for k in range(self.num_agents):
                mb_obs_next[k].append(np.copy(obs[k]))

            re_obs = self.obs
            re_actions = self.actions
            re_obs_next = obs
            re_path_prob = self.model.get_log_action_prob_step(re_obs, re_actions)  # [num_agent, nenv, 1]
            re_actions_onehot = [multionehot(re_actions[k], self.n_actions[k]) for k in range(self.num_agents)]

            # get reward from discriminator
            if self.disc_type == 'decentralized':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))
            elif self.disc_type == 'decentralized-all':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs, axis=1),
                                                                               np.concatenate(re_actions_onehot, axis=1),
                                                                               np.concatenate(re_obs_next, axis=1),
                                                                               re_path_prob[k],
                                                                               discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs, axis=1),
                                                                               np.concatenate(re_actions_onehot, axis=1),
                                                                               np.concatenate(re_obs_next, axis=1),
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))
            else:
                assert False

            for k in range(self.num_agents):
                mb_rewards[k].append(rewards[k])
                mb_report_rewards[k].append(report_rewards[k])
                # mb_rewards[k].append([rewards[k]])
                # mb_report_rewards[k].append([report_rewards[k]])

                """
                tf = np.array(['n' in info[i].keys() for i in range(len(info))])
                if tf.all():
                    mb_is_goal[k].append([info[i]['n'][k]['isGoal'] for i in range(len(info))])
                    mb_is_collision[k].append([info[i]['n'][k]['isCollision'] for i in range(len(info))])
                else:
                    mb_is_goal[k].append([[False, False] for _ in range(self.nenv)])
                    mb_is_collision[k].append([[False, False] for _ in range(self.nenv)])"""
           

            self.states = states
            self.dones = dones
            self.update_obs(obs)
            self.info = info

            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])

                is_goal = []
                is_collision = []
                for l in range(len(self.info)):
                    if self.dones[0][l] and self.dones[1][l]:
                        is_goal.append(True)
                        is_collision.append(False)
                    elif not 'n' in self.info[l].keys():
                        is_goal.append(False)
                        is_collision.append(False)
                    else:
                        is_goal.append(self.info[l]['n'][k]['isGoal'])
                        is_collision.append(self.info[l]['n'][k]['isCollision'])
                mb_is_goal[k].append(is_goal)
                mb_is_collision[k].append(is_collision)
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])

        # batch of steps to batch of rollouts
        traj_obs = [[] for _ in range(self.num_agents)]
        traj_obs_next = [[] for _ in range(self.num_agents)]

        for k in range(self.num_agents):
            traj_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0)
            traj_nobs = traj_obs[k].copy()
            traj_nobs[:-1] = traj_obs[k][1:]
            traj_nobs[-1] = traj_obs[k][0]
            traj_obs_next[k] = traj_nobs.copy()
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_obs_next[k] = np.asarray(mb_obs_next[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_report_rewards[k] = np.asarray(mb_report_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]
            mb_is_goal[k] = np.asarray(mb_is_goal[k], dtype=np.bool).swapaxes(1, 0)
            mb_is_collision[k] = np.asarray(mb_is_collision[k], dtype=np.bool).swapaxes(1, 0)
        #print(mb_obs[0])
        #print(traj_obs[0][0])
        #print(mb_is_goal[0][0])
        for k in range(self.num_agents):
            ep_rews = []
            cols = []
            goals = []
            steps = []
            for t in range(len(traj_obs)):
                traj_len = [np.where(mb_dones[k2][t])[0] for k2 in range(self.num_agents)]
                traj_len = [traj_len[k2] - np.concatenate([[-1], traj_len[k2][:-1]]) for k2 in range(self.num_agents)]
                sprev = 0
                for idx, s in enumerate(np.where(mb_dones[k][t])[0]):
                    traj_tf = np.array([False for _ in range(len(mb_dones[k][t]))])
                    traj_tf[sprev:s+1] = True
                    """print(f't:{t}, tf:{traj_tf}')
                    print(f'tf shape: {traj_tf.shape}')
                    print(mb_is_collision[k])
                    print(f'col shape: {mb_is_collision[k].shape}')
                    print(f'dones shape: {mb_dones[k].shape}')
                    print(f'true_rewards shape: {mb_true_rewards[k].shape}')
                    print(f'is_goal shape: {mb_is_goal[k].shape}')
                    print(f'traj_obs shape: {traj_obs[k].shape}')
                    print(mb_dones[k])"""
                    mb_is_collision[k][t][traj_tf][-1] = False # 最後がTrueになっちゃうので強制的に変更
                    #traj_obs_round = [[round(num[0], 2), round(num[1], 2)]  for num in traj_obs[k][t][tf]] 
                    #print(traj_obs_round)
                    #print(f't, k, sliceNum, length(length2): {t}, {k}, {s}, {np.sum(traj_tf)}({traj_len[k][idx]})')
                    ep_rew = np.sum(np.array(mb_true_rewards[k][t][traj_tf]))
                    goal = np.sum(np.array(mb_is_goal[k][t][traj_tf]))
                    col = np.sum(np.array(mb_is_collision[k][t][traj_tf]))
                    if ep_rew==np.nan:
                        print("ep_rew is nan!!!!")
                    else:
                        #print(f"ep_rew{k}: {ep_rew}, goal:{goal}, col{col}")
                        ep_rews.append(ep_rew)
                        cols.append(col)
                        goals.append(goal)
                        steps.append(traj_len[k][idx])
                    if idx>=len(traj_len[0]) or idx>=len(traj_len[1]):
                        continue

                    if traj_len[0][idx] < GOAL_STEP_THRESHOLD and traj_len[1][idx] < GOAL_STEP_THRESHOLD: # if agent reached goal, they archive thier info

                        traj = traj_obs[k][t][traj_tf]
                        traj_obs_round = [[round(num[0], 2), round(num[1], 2)]  for num in traj] 
                        arc_indi_obs[k].append(traj_obs_round)
                        arc_indi_actions[k].append(multionehot(np.copy(mb_actions[k][t][traj_tf]), self.n_actions[k]).tolist())

                        arc_indi_values[k].append((mb_values[k][t][traj_tf]).tolist())
                        traj_obs_next_round = [[round(num[0], 2), round(num[1], 2)]  for num in traj_obs_next[k][t][traj_tf]] 
                        arc_indi_obs_next[k].append(traj_obs_next_round)
                        arc_indi_scores[k].append(ep_rew)
                        col = np.sum(np.array(mb_is_collision[k][t][traj_tf]))
                        #if not any(mb_true_rewards[k][t]<=-100): # if agent reached goal without collision, they archive info
                        if col==0: # if agent reached goal without collision, they archive info
                            #arc_coop_obs[k].append((traj_obs[k][t]).tolist()
                            arc_coop_obs[k].append(traj_obs_round)
                            arc_coop_actions[k].append(multionehot(np.copy(mb_actions[k][t][traj_tf]), self.n_actions[k]).tolist())
                            arc_coop_values[k].append((mb_values[k][t][traj_tf]).tolist())
                            arc_coop_obs_next[k].append(traj_obs_next_round)
                            arc_coop_scores[k].append(ep_rew)
                    sprev = s+1


            logger.record_tabular('true reward %d' % k, float(np.mean(ep_rews)))
            logger.record_tabular('collision %d' % k, float(np.mean(cols)))
            logger.record_tabular('goal %d' % k, float(np.mean(goals)))
            logger.record_tabular('steps %d' % k, float(np.mean(steps)))
        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_report_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, report_rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_report_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                report_rewards = report_rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    report_rewards = discount_with_dones(report_rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    report_rewards = discount_with_dones(report_rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_report_returns[k][n] = report_rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_report_returns[k] = mb_report_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [multionehot(mb_actions[k], self.n_actions[k]) for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mb_all_nobs = np.concatenate(mb_obs_next, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        if self.nobs_flag:
            return mb_obs, mb_obs_next, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions, \
                   mb_values, mb_all_obs, mb_all_nobs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns, \
                   arc_indi_obs, arc_indi_obs_next, arc_indi_actions, arc_indi_values, arc_indi_all_obs, arc_indi_scores,\
                   arc_coop_obs, arc_coop_obs_next, arc_coop_actions, arc_coop_values, arc_coop_all_obs,arc_coop_scores, mb_is_goal, mb_is_collision
        else:
            return mb_obs, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions,\
                   mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns,\
                   arc_indi_obs, arc_indi_obs_next, arc_indi_actions, arc_indi_values, arc_indi_all_obs, arc_indi_scores,\
                   arc_coop_obs, arc_coop_obs_next, arc_coop_actions, arc_coop_values, arc_coop_all_obs,arc_coop_scores, mb_is_goal, mb_is_collision


def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=1000, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, l2=0.1, d_iters=1, rew_scale=0.1):
    tf.reset_default_graph()
    set_global_seeds(seed)
    buffer = None
    archive_indi_buffer = None
    archive_coop_buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space
    #ac_space = env.action_space
    print(f'行動空間を強制的に４に変更します。これは，5番目の行動（Stop）をしないようにするためです。:')
    ac_space = [spaces.Discrete(4) for _ in range(2)]
    num_agents = (len(ob_space))
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized' or disc_type == 'decentralized-all':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space,
                          state_only=False, discount=gamma, nstack=nstack, index=k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr, l2_loss_ratio=l2) for k in range(num_agents)
        ]
    else:
        assert False

    # add reward regularization
    if env_id == 'simple_tag':
        reward_reg_loss = tf.reduce_mean(
            tf.square(discriminator[0].reward + discriminator[3].reward) +
            tf.square(discriminator[1].reward + discriminator[3].reward) +
            tf.square(discriminator[2].reward + discriminator[3].reward)
        ) + rew_scale * tf.reduce_mean(
            tf.maximum(0.0, 1 - discriminator[0].reward) +
            tf.maximum(0.0, 1 - discriminator[1].reward) +
            tf.maximum(0.0, 1 - discriminator[2].reward) +
            tf.maximum(0.0, discriminator[3].reward + 1)
        )
        reward_reg_lr = tf.placeholder(tf.float32, ())
        reward_reg_optim = tf.train.AdamOptimizer(learning_rate=reward_reg_lr)
        reward_reg_train_op = reward_reg_optim.minimize(reward_reg_loss)

    tf.global_variables_initializer().run(session=model.sess)
    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type,
                    nobs_flag=True)
    nbatch = nenvs * nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    for _ in range(bc_iters):
        e_obs, e_actions, e_nobs, _, _ = expert.get_next_batch(nenvs * nsteps)
        e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
        #lld_loss = model.clone(e_obs, e_a)


    mb_arc_indi_obs = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_indi_actions = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_indi_obs_next = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_indi_all_obs = [[] for _ in range(MAX_ARC)] 
    mb_arc_indi_values = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_indi_scores = [np.ones(MAX_ARC)*-1000 for _ in range(num_agents)]
    mb_arc_coop_obs = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_coop_actions = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_coop_obs_next = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_coop_all_obs = [[] for _ in range(MAX_ARC)]
    mb_arc_coop_values = [[[] for _ in range(MAX_ARC)] for _ in range(num_agents)]
    mb_arc_coop_scores = [np.ones(MAX_ARC)*-1000 for _ in range(num_agents)]
    archive_indi_num = np.zeros(num_agents)
    archive_coop_num = np.zeros(num_agents)


    update_policy_until = 10

    for update in range(1, total_timesteps // nbatch + 1):
        obs, obs_next, states, rewards, report_rewards, masks, actions, values, all_obs, all_nobs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns,\
        arc_indi_obs, arc_indi_obs_next, arc_indi_actions, arc_indi_values, arc_indi_all_obs,arc_indi_scores,\
        arc_coop_obs, arc_coop_obs_next, arc_coop_actions, arc_coop_values, arc_coop_all_obs, arc_coop_scores, mh_is_goal, mh_is_collision = runner.run()

        total_loss = np.zeros((num_agents, d_iters))

        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_obs_next = [obs_next[k][idxs] for k in range(num_agents)]
        mh_values = [values[k][idxs] for k in range(num_agents)]

        if buffer:
            buffer.update(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values)
        else:
            buffer = Dset(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values, randomize=True, num_agents=num_agents,
                          nobs_flag=True)


        for k in range(num_agents):
            if not arc_indi_obs[k]: continue
            for t in range(len(arc_indi_obs[k])):
                mb_arc_indi_obs[k], _ = push_arc(mb_arc_indi_obs[k], arc_indi_obs[k][t], mb_arc_indi_scores[k], arc_indi_scores[k][t])

                mb_arc_indi_actions[k], _ = push_arc(mb_arc_indi_actions[k], arc_indi_actions[k][t], mb_arc_indi_scores[k], arc_indi_scores[k][t])
                mb_arc_indi_obs_next[k], _ = push_arc(mb_arc_indi_obs_next[k], arc_indi_obs_next[k][t], mb_arc_indi_scores[k], arc_indi_scores[k][t])
                mb_arc_indi_values[k], mb_arc_indi_scores[k] = push_arc(mb_arc_indi_values[k], arc_indi_values[k][t], mb_arc_indi_scores[k], arc_indi_scores[k][t])
                archive_indi_num[k] += len(arc_indi_obs[k])

            if not arc_coop_obs[k]: continue
            for t in range(len(arc_coop_obs[k])):
                mb_arc_coop_obs[k], _ = push_arc(mb_arc_coop_obs[k], arc_coop_obs[k][t], mb_arc_coop_scores[k], arc_coop_scores[k][t])

                mb_arc_coop_actions[k], _ = push_arc(mb_arc_coop_actions[k], arc_coop_actions[k][t], mb_arc_coop_scores[k], arc_coop_scores[k][t])
                mb_arc_coop_obs_next[k], _ = push_arc(mb_arc_coop_obs_next[k], arc_coop_obs_next[k][t], mb_arc_coop_scores[k], arc_coop_scores[k][t])
                mb_arc_coop_values[k], mb_arc_coop_scores[k] = push_arc(mb_arc_coop_values[k], arc_coop_values[k][t], mb_arc_coop_scores[k], arc_coop_scores[k][t])
                archive_coop_num[k] += len(arc_coop_obs[k])
        
        archived = [False, False]
        if all(archive_indi_num>0):
            archived[0] = True
            obs_flat = [[] for k in range(num_agents)]
            actions_flat = [[] for k in range(num_agents)]
            obs_next_flat = [[] for k in range(num_agents)]
            values_flat = [[] for k in range(num_agents)]

            for k in range(num_agents):
                for i in range(len(mb_arc_indi_obs[k])):
                    obs_flat[k] += mb_arc_indi_obs[k][i]
                    actions_flat[k] += mb_arc_indi_actions[k][i]
                    obs_next_flat[k] += mb_arc_indi_obs_next[k][i]
                    values_flat[k] += mb_arc_indi_values[k][i]
                obs_flat[k] = np.array(obs_flat[k])    
                actions_flat[k] = np.array(actions_flat[k])    
                obs_next_flat[k] = np.array(obs_next_flat[k])    
                values_flat[k] = np.array(values_flat[k])    

            min_len = np.min([len(obs_flat[k]) for k in range(num_agents)])
            for k in range(num_agents):
                obs_flat[k] = obs_flat[k][0:min_len]
                actions_flat[k] = actions_flat[k][0:min_len]
                obs_next_flat[k] = obs_next_flat[k][0:min_len]
                values_flat[k] = values_flat[k][0:min_len]

            archive_indi_buffer = Dset(obs_flat, actions_flat, obs_next_flat, [], values_flat, randomize=True, num_agents=num_agents, nobs_flag=True)


        if all(archive_coop_num>0):
            archived[1] = True
            archived[0] = True
            obs_flat = [[] for k in range(num_agents)]
            actions_flat = [[] for k in range(num_agents)]
            obs_next_flat = [[] for k in range(num_agents)]
            values_flat = [[] for k in range(num_agents)]

            for k in range(num_agents):
                for i in range(len(mb_arc_coop_obs[k])):
                    obs_flat[k] += mb_arc_coop_obs[k][i]
                    actions_flat[k] += mb_arc_coop_actions[k][i]
                    obs_next_flat[k] += mb_arc_coop_obs_next[k][i]
                    values_flat[k] += mb_arc_coop_values[k][i]
                obs_flat[k] = np.array(obs_flat[k])    
                actions_flat[k] = np.array(actions_flat[k])    
                obs_next_flat[k] = np.array(obs_next_flat[k])    
                values_flat[k] = np.array(values_flat[k])    

            min_len = np.min([len(obs_flat[k]) for k in range(num_agents)])
            for k in range(num_agents):
                obs_flat[k] = obs_flat[k][0:min_len]
                actions_flat[k] = actions_flat[k][0:min_len]
                obs_next_flat[k] = obs_next_flat[k][0:min_len]
                values_flat[k] = values_flat[k][0:min_len]

            archive_coop_buffer = Dset(obs_flat, actions_flat, obs_next_flat, [], values_flat, randomize=True, num_agents=num_agents, nobs_flag=True)

        d_minibatch = nenvs * nsteps
        d_minibatch_quarter = int(d_minibatch/4)
        d_minibatch_half = d_minibatch_quarter*2 
        d_minibatch = d_minibatch_half*2

        for d_iter in range(d_iters):
            if archived[1]:
                expert_batch = [0, 0, d_minibatch]
            elif archived[0]:
                expert_batch = [d_minibatch_half, d_minibatch_half, 0]
            else:
                expert_batch = [d_minibatch, 0, 0]
                
            if expert_batch[0]!=0: 
                e_obs, e_actions, e_nobs, e_all_obs, _ = expert.get_next_batch(expert_batch[0])
            else:
                e_obs = None
            g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch)
            print(f'[exp, indi, coop] = {expert_batch}')

            while(True):
                if not archived[0] or expert_batch[1]==0: break 
                e1_obs, e1_actions, e1_nobs, e1_all_obs, _ = archive_indi_buffer.get_next_batch(expert_batch[1])
                if e_obs:
                    e_obs = [np.concatenate([e_obs[k], e1_obs[k]]) for k in range(num_agents)]
                    e_actions = [np.concatenate([e_actions[k], e1_actions[k]]) for k in range(num_agents)]
                    e_nobs = [np.concatenate([e_nobs[k], e1_nobs[k]]) for k in range(num_agents)]
                else:
                    e_obs = e1_obs
                    e_actions = e1_actions
                    e_nobs = e1_nobs
                batch_size = expert_batch[0] + expert_batch[1] 
                if len(e_obs[0])>=batch_size:
                    e_obs = [e_obs[k][0:batch_size] for k in range(num_agents)]
                    e_actions = [e_actions[k][0:batch_size] for k in range(num_agents)]
                    e_nobs = [e_nobs[k][0:batch_size] for k in range(num_agents)]
                    break
                
            while(True):
                if not archived[1] or expert_batch[2]==0: break 
                e1_obs, e1_actions, e1_nobs, e1_all_obs, _ = archive_coop_buffer.get_next_batch(expert_batch[2])
                if e_obs:
                    e_obs = [np.concatenate([e_obs[k], e1_obs[k]]) for k in range(num_agents)]
                    e_actions = [np.concatenate([e_actions[k], e1_actions[k]]) for k in range(num_agents)]
                    e_nobs = [np.concatenate([e_nobs[k], e1_nobs[k]]) for k in range(num_agents)]
                else:
                    e_obs = e1_obs
                    e_actions = e1_actions
                    e_nobs = e1_nobs
                batch_size = expert_batch[0] + expert_batch[1] + expert_batch[2]
                if len(e_obs[0])>=batch_size:
                    e_obs = [e_obs[k][0:batch_size] for k in range(num_agents)]
                    e_actions = [e_actions[k][0:batch_size] for k in range(num_agents)]
                    e_nobs = [e_nobs[k][0:batch_size] for k in range(num_agents)]
                    break
                

            e_obs2 = []
            e_nobs2 = []
            g_obs2 = []
            g_nobs2 = []
            for k in range(num_agents):
                e_obs_round = [[round(num[0], 2), round(num[1], 2)] for num in e_obs[k]] 
                e_obs2.append(e_obs_round)

                e_nobs_round = [[round(num[0], 2), round(num[1], 2)] for num in e_nobs[k]]
                e_nobs2.append(e_nobs_round)

                g_obs_round = [[round(num[0], 2), round(num[1], 2)] for num in g_obs[k]]
                g_obs2.append(g_obs_round)

                g_nobs_round = [[round(num[0], 2), round(num[1], 2)] for num in g_nobs[k]]
                g_nobs2.append(g_nobs_round)

            e_obs = np.array(e_obs2)
            e_nobs = np.array(e_nobs2)
            g_obs = np.array(g_obs2)
            g_nobs = np.array(g_nobs2)

            e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
            g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

            g_log_prob = model.get_log_action_prob(g_obs, g_a)
            e_log_prob = model.get_log_action_prob(e_obs, e_a)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs[k],
                        g_actions[k],
                        g_nobs[k],
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs[k],
                        e_actions[k],
                        e_nobs[k],
                        e_log_prob[k].reshape([-1, 1])
                    )
            elif disc_type == 'decentralized-all':
                g_obs_all = np.concatenate(g_obs, axis=1)
                g_actions_all = np.concatenate(g_actions, axis=1)
                g_nobs_all = np.concatenate(g_nobs, axis=1)
                e_obs_all = np.concatenate(e_obs, axis=1)
                e_actions_all = np.concatenate(e_actions, axis=1)
                e_nobs_all = np.concatenate(e_nobs, axis=1)
                for k in range(num_agents):
                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs_all,
                        g_actions_all,
                        g_nobs_all,
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs_all,
                        e_actions_all,
                        e_nobs_all,
                        e_log_prob[k].reshape([-1, 1])
                    )
            else:
                assert False

            if env_id == 'simple_tag':
                if disc_type == 'decentralized':
                    feed_dict = {discriminator[k].obs: np.concatenate([g_obs[k], e_obs[k]], axis=0)
                                 for k in range(num_agents)}
                elif disc_type == 'decentralized-all':
                    feed_dict = {discriminator[k].obs: np.concatenate([g_obs_all, e_obs_all], axis=0)
                                 for k in range(num_agents)}
                else:
                    assert False
                feed_dict[reward_reg_lr] = discriminator[0].lr.value()
                model.sess.run(reward_reg_train_op, feed_dict=feed_dict)

            idx += 1

        if update > update_policy_until:  # 10 policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                if update > update_policy_until:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    logger.record_tabular("archive_indi_num %d" %k, int(archive_indi_num[k]))
                    logger.record_tabular("archive_coop_num %d" %k, int(archive_coop_num[k]))
                    logger.record_tabular("archive_indi_scores %d" %k, np.mean(mb_arc_indi_scores[k]))
                    logger.record_tabular("archive_coop_scores %d" %k, np.mean(mb_arc_coop_scores[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
                    except:
                        pass
                    #logger.record_tabular('true reward %d' % k, float(np.mean(mh_true_returns[k])))
                    #ep_rew = np.mean(np.sum(np.array(mh_true_rewards[k][:]), axis=1))
                    #logger.record_tabular('Exp. Ret %d' % k, float(np.sum(mh_true_rewards[k])/(nenvs * len(mh_true_rewards[k][0])/50)))

            total_loss_m = np.mean(total_loss, axis=1)
            for k in range(num_agents):
                logger.record_tabular("total_loss %d" % k, total_loss_m[k])
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            pkl.dump(mb_arc_indi_obs, open(osp.join(logger.get_dir(), f'archive_indi_obs{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_indi_actions, open(osp.join(logger.get_dir(), f'archive_indi_actions{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_indi_obs_next, open(osp.join(logger.get_dir(), f'archive_indi_obs_next{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_indi_scores, open(osp.join(logger.get_dir(), f'archive_indi_scores{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_coop_obs, open(osp.join(logger.get_dir(), f'archive_coop_obs{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_coop_actions, open(osp.join(logger.get_dir(), f'archive_coop_actions{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_coop_obs_next, open(osp.join(logger.get_dir(), f'archive_coop_obs_next{update:05}.pkl'), 'wb'))
            pkl.dump(mb_arc_coop_scores, open(osp.join(logger.get_dir(), f'archive_coop_scores{update:05}.pkl'), 'wb'))
            if disc_type == 'decentralized' or disc_type == 'decentralized-all':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            else:
                assert False
    coord.request_stop()
    # coord.join(enqueue_threads)
