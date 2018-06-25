DDPG_DEFAULT_PARAMS = {
    # env
    'env_id': 'HalfCheetah-v2',

    # ddpg
    'layer_norm': True,
    'render': False,
    'normalize_returns':False,
    'normalize_observations':True,
    'actor_lr': 0.0001,  # critic learning rate
    'critic_lr': 0.001,  # actor learning rate
    'critic_l2_reg': 1e-2,
    'popart': False,
    'gamma': 0.99,

    # training
    'seed': 0,
    'nb_epochs':500, # number of epochs
    'nb_epoch_cycles': 20,  # per epoch
    'nb_rollout_steps': 100,  # sampling batches per cycle
    'nb_train_steps': 100,  # training batches per cycle
    'batch_size': 64,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'reward_scale': 1.0,
    'clip_norm': None,

    # exploration
    'noise_type':'adaptive-param_0.2',

    # debugging, logging and visualization
    'render_eval': False,
    'nb_eval_steps':100,
    'evaluation':False,
}

ENSEMBLE_DDPG_DEFAULT_PARAMS = {
    # env
    'env_id': 'HalfCheetah-v2',

    # ensemble_ddpg
    'layer_norm': False,
    'render': False,
    'normalize_returns': False,
    'normalize_observations': False,
    'actor_lr': 0.001,  # critic learning rate
    'critic_lr': 0.001,  # actor learning rate
    'critic_l2_reg': 0.0,
    'popart': False,
    'gamma': 0.99,

    # training
    'seed': 0,
    'nb_epochs': 500,    # number of epochs
    'nb_epoch_cycles': 20,  # per epoch
    'nb_rollout_steps': 100,  # sampling batches per cycle
    'nb_train_steps': 100,  # training batches per cycle
    'batch_size': 64,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'reward_scale': 1.0,
    'clip_norm': None,

    # exploration
    'noise_type': 'normal_0.1',

    # debugging, logging and visualization
    'render_eval': False,
    'nb_eval_steps': 100,
    'evaluation': False,
}


TD3_DEFAULT_PARAMS = {
    # env
    'env_id': 'HalfCheetah-v2',

    # ddpg
    'layer_norm': True,
    'render': False,
    'normalize_returns':False,
    'normalize_observations':True,
    'actor_lr': 0.001,  # critic learning rate
    'critic_lr': 0.001,  # actor learning rate
    'critic_l2_reg': 0.0,
    'popart': False,
    'gamma': 0.99,
    'network': "TD3",
    # TD3 is the network proposed in that paper, and oldDDPG is the
    # original architecture in openai baseline
    

    # training
    'seed': 0,
    'nb_epochs': 500,   # number of epochs
    'nb_epoch_cycles': 20,  # per epoch
    'nb_rollout_steps': 100,  # sampling batches per cycle
    'nb_train_steps': 100,  # training batches per cycle
    'batch_size': 100,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'reward_scale': 1.0,
    'clip_norm': None,

    # exploration
    'noise_type':'normal_0.1',

    # debugging, logging and visualization
    'render_eval': False,
    'nb_eval_steps':100,
    'evaluation':False,
}

