import os
import socket
import torch
import torch.distributed as dist


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_on_master(obj, path: str):
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)
