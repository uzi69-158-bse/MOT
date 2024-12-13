import os
import psutil

def monitor_memory_usage():
    """Monitor NUMA memory usage and log it."""
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()

    print(f"Process Memory Usage: RSS: {mem_info.rss / (1024 ** 2)} MB, VMS: {mem_info.vms / (1024 ** 2)} MB")
    # Additional NUMA-specific memory checks can be added here
