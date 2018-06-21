from copy import copy
# use old copy ? 
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from mpi4py import MPI



def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class TD3(object):
    def __init__(self, actor, critic0, critic1, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.005, normalize_returns=False, enable_popart=False, normalize_observations=False,
        batch_size=100, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
        action_noise_scale=0.2, action_noise_clip=0.5,
        critic_l2_reg=0., actor_lr=1e-3, critic_lr=1e-3, clip_norm=None, reward_scale=1., use_mpi_adam=False):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range

        self.use_mpi_adam = use_mpi_adam

        # set the primary critic to critic0, and set supplementary critic as critic1
        self.critic0 = critic0
        self.critic1 = critic1

        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # remember the noise scale and clip
        self.action_noise_scale = action_noise_scale
        self.action_noise_clip = action_noise_clip

        # Observation normalization.

        # TODO: did td3 used normalization ?
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        # TODO: what's the performance when we're normalizing the neurons ?
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor

        # set up different target critics, primary and supplementary
        target_critic0 = copy(critic0)
        target_critic0.name = 'target_critic0'
        self.target_critic0 = target_critic0
        target_critic1 = copy(critic1)
        target_critic1.name = 'target_critic1'
        self.target_critic1 = target_critic1

        # Create networks and core TF parts that are shared across setup parts.
        
        # actor_tf pi(s) is built from the actor and normalized_obs0 
        self.actor_tf = actor(normalized_obs0)

        # normalized_critic_tf normalized Q(s,a) is built from the observation and action 
        # two normalized critic functions Q0(s,a) and Q1(s,a)
        self.normalized_critic_tf0 = critic0(normalized_obs0, self.actions)
        self.normalized_critic_tf1 = critic1(normalized_obs0, self.actions)

        # critic_tf Q(s,a) is built from de-normalization and clipping from normalized Q(s,a)
        # two critic functions Q0(s,a) and Q1(s,a)
        self.critic_tf0 = denormalize(tf.clip_by_value(self.normalized_critic_tf0, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.critic_tf1 = denormalize(tf.clip_by_value(self.normalized_critic_tf1, self.return_range[0], self.return_range[1]), self.ret_rms)

        # normalized_critic_with_actor_tf normalized Q(s,pi(s)) is built from the observation, and action provied by actor 
        self.normalized_critic_with_actor_tf0 = critic0(normalized_obs0, self.actor_tf, reuse=True)
        self.normalized_critic_with_actor_tf1 = critic1(normalized_obs0, self.actor_tf, reuse=True)

        # critic_with_actor_tf is built from de-normalization and clipping from normalized Q(s,pi(s))
        # original scale Q0(s,pi(s)) Q1(s,pi(s))
        self.critic_with_actor_tf0 = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor_tf0, self.return_range[0], self.return_range[1]),
            self.ret_rms)
        self.critic_with_actor_tf1 = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor_tf1, self.return_range[0], self.return_range[1]),
            self.ret_rms)

        # Q_obs1 Q(s',pi'(s)) is built from next state s'(observation), target actor pi', and denormalization  

        target_action = target_actor(normalized_obs1)
        target_action_noise = tf.clip_by_value(tf.random_normal(
            tf.shape(target_action), mean=0.0, stddev=action_noise_scale,dtype=tf.float32),
            clip_value_min=-action_noise_clip,clip_value_max=action_noise_clip)
        noisy_target_action = tf.clip_by_value(target_action + target_action_noise,
                                               clip_value_min=action_range[0],clip_value_max=action_range[1])
        Q_obs1_val0 = denormalize(target_critic0(normalized_obs1, noisy_target_action), self.ret_rms)
        Q_obs1_val1 = denormalize(target_critic1(normalized_obs1, noisy_target_action), self.ret_rms)

        Q_obs = tf.reshape(tf.reduce_min(tf.concat([Q_obs1_val0,Q_obs1_val1],axis=1),axis=1),shape=[-1,1])

        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs
        self.target_Q0 = self.target_Q1 = self.target_Q

        print("[Tiancheng] Shape of target Q",self.target_Q.shape)

        # merge trainable variables into one set
        self.target_critic_vars = target_critic0.vars + target_critic1.vars
        self.critic_vars = self.critic0.vars + self.critic1.vars
        self.critic_trainable_vars = self.critic0.trainable_vars + self.critic1.trainable_vars
        print("[Tiancheng] Re-asure Trainable Vars")
        print("[Tiancheng] Critic 0 Trainable Vars", self.critic0.trainable_vars)
        print("[Tiancheng] Critic 1 Trainable Vars", self.critic1.trainable_vars)


        # Set up parts.
        if self.param_noise is not None:
            # TODO: what's param noise ? 
            self.setup_param_noise(normalized_obs0)

        print("[Tiancheng] Before Setup, Len Actor and Target Actor", len(self.actor.trainable_vars), len(self.target_actor.vars))

        # setup optimizer
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()

        # TODO: what's popart ?
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()

        print("[Tiancheng] Len Actor and Target Actor",len(self.actor.trainable_vars),len(self.target_actor.vars))

        self.setup_target_network_updates()

    def setup_target_network_updates(self):
        if self.use_mpi_adam:
            actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars,
                                                                        self.tau)
            critic_init_updates, critic_soft_updates = get_target_updates(self.critic_vars, self.target_critic_vars,
                                                                          self.tau)
            self.target_init_updates = [actor_init_updates, critic_init_updates]
            self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

            self.target_soft_update_actor = actor_soft_updates
            self.target_soft_update_critic = critic_soft_updates
        else:
            actor_init_updates, actor_soft_updates = get_target_updates(self.actor.trainable_vars,
                                                                        self.target_actor.vars, self.tau)
            critic_init_updates, critic_soft_updates = get_target_updates(self.critic_trainable_vars,
                                                                          self.target_critic_vars, self.tau)
            self.target_init_updates = [actor_init_updates, critic_init_updates]
            self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

            self.target_soft_update_actor = actor_soft_updates
            self.target_soft_update_critic = critic_soft_updates

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')

        # Here use the Q(s,pi(s)) as the loss function
        #   use primary critic function to generate policy updates
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf0)

        # get actor shapes ? ( for what ? )
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        
        # TODO: not sure what happens here . 
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        if self.use_mpi_adam:
            self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
                beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            self.actor_grads = list(
                zip(tf.gradients(self.actor_loss, self.actor.trainable_vars), self.actor.trainable_vars))

            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr,beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.actor_train = self.actor_optimizer.apply_gradients(self.actor_grads)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')

        # normalize critc target, normalized y
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.target_Q, self.ret_rms), self.return_range[0], self.return_range[1])
        normalized_critic_target_tf = tf.stop_gradient(normalized_critic_target_tf)

        # Use square error between normalized_critic_tf normalized Q(s,a) and normalized critic_target y
        # ( not use denormalized version ) as loss function, for two different critic, we need to train them both
        self.critic_loss0 = tf.reduce_mean(tf.square(self.normalized_critic_tf0 - normalized_critic_target_tf))
        self.critic_loss1 = tf.reduce_mean(tf.square(self.normalized_critic_tf1 - normalized_critic_target_tf))

        # merge two process as one pass
        self.critic_loss = self.critic_loss0 + self.critic_loss1

        # apply l2_regularization on some trainable variables and add them into loss function
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic_trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))

            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))

            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        # get critic parameter shapes ?  And reduce something. ( TODO )
        critic_shapes = [var.get_shape().as_list() for var in self.critic_trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic_trainable_vars, clip_norm=self.clip_norm)
        if self.use_mpi_adam :
            self.critic_optimizer = MpiAdam(var_list=self.critic_trainable_vars,
                                            beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            self.critic_grads = list(zip(tf.gradients(self.critic_loss,self.critic_trainable_vars),self.critic_trainable_vars))
            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr,beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.critic_train = self.critic_optimizer.apply_gradients(self.critic_grads)

    def setup_popart(self):
        # TODO: important, what's popart and what should we do for it ?
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic0.output_vars + self.critic1.output_vars,
                   self.target_critic0.output_vars + self.target_critic1.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf0)]
        names += ['reference_Q0_mean']
        ops += [reduce_std(self.critic_tf0)]
        names += ['reference_Q0_std']

        ops += [tf.reduce_mean(self.critic_tf1)]
        names += ['reference_Q1_mean']
        ops += [reduce_std(self.critic_tf1)]
        names += ['reference_Q1_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf0)]
        names += ['reference_actor_Q0_mean']
        ops += [reduce_std(self.critic_with_actor_tf0)]
        names += ['reference_actor_Q0_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf1)]
        names += ['reference_actor_Q1_mean']
        ops += [reduce_std(self.critic_with_actor_tf1)]
        names += ['reference_actor_Q1_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    # compute the action from the observation pi(s)
    #   has an option to compute the q function at the same time 
    def pi(self, obs, apply_noise=True, compute_Q=False):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs]}
        if compute_Q:
            # TODO: not sure what to do for this critic_with_actor_tf, set to critic_with_actor_tf0
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf0], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self, take_update=True):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        # if self.normalize_returns and self.enable_popart:
        #     # compute old mean, old std and target Q values
        #     # old mean and std is used for normalization
        #     # and target Q values for
        #     old_mean, old_std, Q0, Q1, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q0,self.target_Q1,self.target_Q], feed_dict={
        #         self.obs1: batch['obs1'],
        #         self.rewards: batch['rewards'],
        #         self.terminals1: batch['terminals1'].astype('float32'),
        #     })
        #
        #     # compute something (TODO)
        #     self.ret_rms.update(target_Q.flatten())
        #     self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
        #         self.old_std : np.array([old_std]),
        #         self.old_mean : np.array([old_mean]),
        #     })
        #
        #     # Run sanity check. Disabled by default since it slows down things considerably.
        #     # print('running sanity check')
        #     # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
        #     #     self.obs1: batch['obs1'],
        #     #     self.rewards: batch['rewards'],
        #     #     self.terminals1: batch['terminals1'].astype('float32'),
        #     # })
        #     # print(target_Q_new, target_Q, new_mean, new_std)
        #     # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        # else:
        #     # compute target Q value functions ( ( 1 - terminal ) * gamma * Q(s,pi(s)) + r )
        #     Q0, Q1, target_Q = self.sess.run([self.target_Q0,self.target_Q1,self.target_Q], feed_dict={
        #         self.obs1: batch['obs1'],
        #         self.rewards: batch['rewards'],
        #         self.terminals1: batch['terminals1'].astype('float32'),
        #     })

        # Get all gradients and perform a "synced update".
        # compute the gradients of actor and critic

        if self.use_mpi_adam:
            ops = [self.critic_grads, self.critic_loss]

            critic_grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

            if take_update:
                ops = [self.actor_grads, self.actor_loss]
                actor_grads, actor_loss = self.sess.run(ops, feed_dict={
                    self.obs0: batch['obs0'],
                })

                self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
                return critic_loss, actor_loss

        else:
            ops = [self.critic_train, self.critic_grads, self.critic_loss]

            _, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            if take_update:
                ops = [self.actor_train, self.actor_grads, self.actor_loss]
                _, actor_grads, actor_loss = self.sess.run(ops, feed_dict={
                    self.obs0: batch['obs0'],
                })
                return critic_loss, actor_loss


        return critic_loss, 0

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        if self.use_mpi_adam:
            self.actor_optimizer.sync()
            self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self,stop_critic_training,stop_actor_training):
        if stop_actor_training:
            self.sess.run(self.target_soft_update_critic)
            return
        if stop_critic_training:
            self.sess.run(self.target_soft_update_actor)
            return
        self.sess.run(self.target_soft_updates)



    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
