# utils/distributed.py

from __future__ import annotations

import os
import torch
import torch.distributed as dist


# ---------------------------------------------------------
# Distributed initialization
# ---------------------------------------------------------
def is_dist_avail_and_initialized() -> bool:
    """Check whether torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return total number of processes participating in training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Return True if the current process is rank 0."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if is_dist_avail_and_initialized():
        dist.barrier()


def init_distributed_mode() -> None:
    """Initialize distributed training environment.

    Supports:
        - torchrun (MASTER_ADDR/MASTER_PORT)
        - Slurm (SLURM_PROCID)
        - Manual env setup

    After initialization, the process's device is set automatically.
    """
    if is_dist_avail_and_initialized():
        return  # Already initialized

    # torchrun style
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", 0))

    # slurm style
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        gpu = rank % torch.cuda.device_count()

    else:
        # Single process, not distributed
        print("[distributed] Not running in distributed mode.")
        return

    torch.cuda.set_device(gpu)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    dist.barrier()
    print(f"[distributed] Initialized: rank={rank}, world_size={world_size}, gpu={gpu}")


# ---------------------------------------------------------
# Rank-safe print
# ---------------------------------------------------------
def print_once(*args, **kwargs) -> None:
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)
