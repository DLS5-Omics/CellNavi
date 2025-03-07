import os
import sys
from pathlib import Path
from cellnavi.trainer import Trainer

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch


def main():
    try:
        # Check the number of GPUs available
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            print(f"CUDA is available. Number of GPUs: {gpu_count}")
        else:
            print("CUDA is available, but no GPUs are detected.")
    except Exception as e:

        print(f"Error checking CUDA availability: {e}")
        
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ["LOCAL_RANK"] = '0'
        os.environ["WORLD_SIZE"] = '1'
        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = '29500'
        
    world_size = int(os.environ["WORLD_SIZE"]) ## the total number of processes in the distributed setup
    world_rank = int(os.environ["RANK"]) ## the global rank of the process within the distributed setup
    local_rank = int(os.environ["LOCAL_RANK"]) ## the rank of the process on the local machine 
    master_ip = os.environ["MASTER_ADDR"] ## the IP address or hostname of the master node (rank 0)
    master_port = os.environ["MASTER_PORT"] ## the port number on which the master node listens for incoming connections from worker processes
    master_uri = "tcp://%s:%s" % (master_ip, master_port)
    

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=master_uri,
        world_size=world_size,
        rank=world_rank,
    )
    torch.cuda.set_device(local_rank)

    with Trainer(local_rank, world_rank) as trainer:
        trainer.train()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
