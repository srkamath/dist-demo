import torch
import logging
import sys
from torch import distributed as dist

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

def main() -> None:
    use_gpu = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_gpu else "gloo")
    dist.barrier()
    logger.info(f"Hello from rank {dist.get_rank()}")
    dist.barrier()

if __name__ == "__main__":
    main()
