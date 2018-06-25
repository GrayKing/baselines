DEFAULT_PARAMS = {
    # env
    'env_id': 'HalfCheetah-v2',  # max absolute value of actions on different coordinates

    # ddpg
    'layer_norm': True,
    'render': False,
    'normalize_returns':False,
    'normalize_observations':True,
    'actor_lr': 0.001,  # critic learning rate
    'critic_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'critic_l2_reg': 1e-2,
    'popart': False,
    'gamma': 0.99,

    # training
    'seed': 0,
    'nb_epochs':500, # number of epochs
    'nb_epoch_cycles': 50,  # per epoch
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


