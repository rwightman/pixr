from typing import Optional

import chugr
from chugr.common import DataArg, DataCfg, DataTaskCfg, DistributedCfg, LoaderBundle


def create_loader(
        data: DataArg,
        task_cfg: DataTaskCfg,
        is_training: bool = False,
        start_interval: int = 0,
        seed: int = 0,
        world_size: int = 1,
        global_rank: int = 0,
        local_rank: int = 0,
) -> LoaderBundle:
    """ Create a chugr loader (webdataset or HF datasets) for the given task data pipeline config. """
    data_cfg = DataCfg.from_arg(data)
    return chugr.create_loader(
        data_cfg,
        task_cfg,
        is_training=is_training,
        start_interval=start_interval,
        seed=seed,
        distributed=DistributedCfg(world_size=world_size, local_rank=local_rank, global_rank=global_rank),
    )
