"""Check that resampled chugr streams differ across training ranks.

Run via torchrun on >= 2 GPUs. Uses the real WDS loader and its workers, retaining
sample keys without decoding images / PDFs or loading a model. Identical rank streams
would silently reduce the effective global training batch size.
"""
import argparse
import os
from collections import Counter

import torch
import torch.distributed as dist
import webdataset as wds

from chugr.common import DistributedCfg, rank_seed
from chugr.wds.loader import create_loader_wds


def sample_identity(sample):
    return {'identity': sample['__url__'] + '::' + sample['__key__']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='WDS shard pattern (quote brace expansions)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--samples', type=int, default=16, help='samples per rank, rounded to full worker batches')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-overlap', type=float, default=0.5,
                        help='fail if two ranks share more than this fraction of their samples')
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world < 2 or args.samples < 1 or args.workers < 0:
        parser.error('requires >= 2 ranks, positive samples and nonnegative workers')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    try:
        torch.manual_seed(rank_seed(args.seed, rank))  # same rank-specific process seeding as the train app
        loader = create_loader_wds(
            args.source,
            task_pipeline=[wds.map(sample_identity)],
            is_training=True,
            num_samples=args.samples * world,
            batch_size=2,
            num_workers=args.workers,
            persistent_workers=False,
            resampled=True,
            seed=args.seed,  # the app intentionally passes the same data seed to all ranks
            distributed=DistributedCfg(world_size=world, global_rank=rank, local_rank=local_rank),
        )
        identities = [key for batch in loader for key in batch['identity']]
        streams = [None] * world
        dist.all_gather_object(streams, identities)
        assert all(streams), 'A rank received no samples.'
        pairs = [(a, b) for a in range(world) for b in range(a + 1, world)]
        duplicate_pairs = [(a, b) for a, b in pairs if streams[a] == streams[b]]
        # fraction of samples shared by two ranks (resampling w/ replacement, some chance overlap is expected)
        overlap = {
            (a, b): sum((Counter(streams[a]) & Counter(streams[b])).values()) / min(len(streams[a]), len(streams[b]))
            for a, b in pairs
        }
        if rank == 0:
            for r, stream in enumerate(streams):
                print(f'rank {r}: {len(stream)} samples; first keys: {stream[:8]}', flush=True)
            print(f'Identical rank streams: {duplicate_pairs}', flush=True)
            print(f'Sample overlap between ranks: { {k: round(v, 3) for k, v in overlap.items()} }', flush=True)
        assert not duplicate_pairs, f'Resampled streams are identical across ranks {duplicate_pairs}.'
        too_similar = {k: v for k, v in overlap.items() if v > args.max_overlap}
        assert not too_similar, f'Resampled streams overlap more than {args.max_overlap} across ranks {too_similar}.'
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
