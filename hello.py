import torch
import logging
import sys
import os
from torch import distributed as dist

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

def main() -> None:
    use_gpu = torch.cuda.is_available()
    local_rank = int(os.environ['LOCAL_RANK'])
    if use_gpu:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_gpu else "gloo", init_method='env://')
    try:
        dist.barrier(device_ids=[local_rank] if use_gpu else None)
        logger.info(f"Hello World from rank {dist.get_rank()}")
        dist.barrier(device_ids=[local_rank] if use_gpu else None)
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
