import json
import os
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
)

from mpi4py import MPI


def run_task(vv, log_dir=None, exp_name=None, allow_extra_parameters=False):
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # load configuration file for running this experiment
    import experiments.config as config

    # Seed for multi-CPU MPI implementation ( rank = 0 for single threaded implementation )
    rank_seed = vv['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    # load params from config
    params = config.get_default_params(vv)

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

    # TODO: according to input parameters, automatically decide which algorithm to use
    if vv.items()['algo'] == "DDPG":
        from baselines.ddpg.main import run
        run(**params)
    elif vv.items()['algo'] == "td3":
        from baselines.td3.main import run
        run(**params)
    else:
        from baselines.ensemble_ddpg.main import run
        run(**params)

