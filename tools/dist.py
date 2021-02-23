import torch
import torch.distributed as dist


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across devices.
    Args:
        tensors (list or tensor): tensors to perform all gather across all processes in
        all devices.
    """

    if not dist.is_initialized():
        return tensors
    list_flag = False
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
        list_flag = True
    cpu_flag = False
    if tensors[0].device == torch.device('cpu'):
        cpu_flag = True
        tensors = [tensor.cuda() for tensor in tensors]
    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    if cpu_flag:
        output_tensor = [tensor.cpu() for tensor in output_tensor]
    if list_flag:
        return output_tensor[0]
    else:
        return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list or tensor): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    if not dist.is_initialized():
        return tensors
    list_flag = False
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
        list_flag = True
    cpu_flag = False
    if tensors[0].device == torch.device('cpu'):
        cpu_flag = True
        tensors = [tensor.cuda() for tensor in tensors]
    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    if cpu_flag:
        tensors = [tensor.cpu() for tensor in tensors]
    if list_flag:
        return tensors[0]
    else:
        return tensors


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
