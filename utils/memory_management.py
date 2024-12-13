import torch

def pin_memory(tensor):
    """Pin memory for faster access during transfers between CPU and GPU."""
    pinned_tensor = tensor.pin_memory()
    return pinned_tensor

def free_memory(tensor):
    """Safely deallocate memory."""
    del tensor
    torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
