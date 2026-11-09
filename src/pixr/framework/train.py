import time

from .task import TaskTrain


def train_one_interval(
        task: TaskTrain,
        loader,
):
    task.interval_start()
    start = time.time()
    num_samples = 0
    for i, sample in enumerate(loader.loader):
        output, loss = task.step(sample)
        num_samples += _batch_size(sample)
        task.after_step(sample, output, loss)
    task.interval_end()
    elapsed = time.time() - start
    return {'samples_per_sec': num_samples * task.device_env.world_size / max(elapsed, 1e-6)}


def _batch_size(sample):
    x = sample['image_input'] if isinstance(sample, dict) else sample[0]
    if isinstance(x, dict):
        x = x['patches']
    return x.shape[0]
