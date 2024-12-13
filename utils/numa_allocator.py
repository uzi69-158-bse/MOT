import torch
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def allocate_numa_memory(size, cpu_id=0):
    """Simulate NUMA-like behavior by setting CPU affinity and pinning memory.
    
    Args:
        size (tuple): Size of the tensor to allocate (e.g., (1024, 1024)).
        cpu_id (int): The CPU core ID to which the process should be pinned.
    
    Returns:
        torch.Tensor: A pinned PyTorch tensor allocated on the specified CPU core.
    
    Raises:
        ValueError: If the size is not a valid tensor shape.
        RuntimeError: If setting CPU affinity fails.
    """
    # Validate the size parameter
    if not isinstance(size, (tuple, list)) or len(size) == 0:
        raise ValueError("Size must be a non-empty tuple or list specifying tensor dimensions.")
    
    try:
        # Set process affinity to the specified CPU core
        process = psutil.Process()
        if cpu_id < psutil.cpu_count(logical=False):            
            process.cpu_affinity([cpu_id])
            logging.info(f"Process pinned to CPU {cpu_id} to simulate NUMA allocation.")
        else:
            logging.warning(f"CPU {cpu_id} is out of range. Using default CPU affinity.")

        # Allocate pinned memory for PyTorch tensor
        tensor = torch.empty(size, dtype=torch.float32, device="cpu")
        logging.info("Memory pinned to improve access speed.")
        return tensor

    except Exception as e:
        logging.error(f"Failed to allocate NUMA memory: {e}")
        raise RuntimeError("Failed to allocate NUMA memory.")

def deallocate_numa_memory(tensor):
    """Deallocate memory by deleting the tensor in PyTorch.
    
    Args:
        tensor (torch.Tensor): The tensor to deallocate.
    """
    if tensor is not None:
        del tensor
        logging.info("Tensor memory deallocated.")
    else:
        logging.warning("Attempted to deallocate a None tensor.")

