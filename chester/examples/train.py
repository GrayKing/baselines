import sys
import json
import time
import os
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

def run(env_id, seed, noise_type, layer_norm, evaluation, buffer_size, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(buffer_size), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def run_task(vv, log_dir=None, exp_name=None, allow_extra_parameters=False):
    override_params = {}
    # Fork for multi-CPU MPI implementation.
    if 'num_cpu' in vv and vv['num_cpu'] > 1:
        whoami = mpi_fork(vv['num_cpu'])
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if log_dir or logger.get_dir() is None:
            logger.configure(dir=log_dir)
    else:
        if log_dir or logger.get_dir() is None:
            logger.configure(dir=log_dir)

    logdir = logger.get_dir()

    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # load configuration file for running this experiment
    import chester.examples.config as config

    # Seed for multi-CPU MPI implementation ( rank == 0 by default )
    rank_seed = vv['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS

    # update all her parameters
    if not allow_extra_parameters:
        for k,v in vv.items():
            if k not in config.DEFAULT_PARAMS:
                print("[ Warning ] Undefined Parameters %s with value %s"%(str(k),str(v)))
        params.update(**{k: v for (k, v) in vv.items() if
                         k in config.DEFAULT_PARAMS})
    else:
        params.update(**{k: v for (k, v) in vv.items()})

    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(params, f)

    run(**params)


