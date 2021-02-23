import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None


def run(
        local_rank, func, arg, cfg, init_method="tcp://localhost:12345", backend='nccl'
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: requiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        arg (Namespace): command line argument
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Initialize the process group.

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=cfg.num_gpus,
        rank=local_rank,
    )

    func(arg, cfg)


def launch_job(func, arg, cfg, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        func (function(CfgNode)): job to run on GPU(s)
        arg (Namespace): command line args
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.num_gpus > 1:
        torch.multiprocessing.spawn(
            run,
            nprocs=cfg.num_gpus,
            args=(
                func,
                arg,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(arg, cfg)
