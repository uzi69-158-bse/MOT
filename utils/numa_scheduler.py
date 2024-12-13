import ctypes
from ctypes import wintypes

# Define ULONG_PTR for compatibility
ULONG_PTR = ctypes.POINTER(wintypes.WPARAM)  # or just use ctypes.c_void_p for a generic pointer

# Constants for logical processor relationships
RelationNumaNode = 0
RelationProcessorCore = 1

class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
    _fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        ("NumaNodeNumber", wintypes.DWORD),
        ("Reserved", wintypes.DWORD * 2),
        ("ProcessorMask", ULONG_PTR),
    ]

def get_cpus_for_numa_node(numa_node):
    """Get the list of CPU core IDs for a specific NUMA node using Windows API.
    
    Args:
        numa_node (int): The NUMA node ID.
    
    Returns:
        list[int]: List of CPU core IDs for the specified NUMA node.
    """
    cpus = []
    buffer_size = ctypes.c_size_t(0)  # Initialize buffer_size as a ctypes instance
    processor_info = None

    # Call to get the required buffer size
    ctypes.windll.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(buffer_size))
    
    # Allocate the buffer
    processor_info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * (buffer_size.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)))()
    
    # Get the processor information
    if ctypes.windll.kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, processor_info, ctypes.byref(buffer_size)):
        for i in range(buffer_size.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)):
            if processor_info[i].Relationship == RelationNumaNode and processor_info[i].NumaNodeNumber == numa_node:
                # Get the CPUs for this NUMA node
                for cpu in range(32):  # Check for CPU bits in the mask
                    if processor_info[i].ProcessorMask & (1 << cpu):
                        cpus.append(cpu)

    return cpus

def display_numa_info(numa_node):
    """Display information about CPUs belonging to a specific NUMA node.
    
    Args:
        numa_node (int): The NUMA node ID to display information for.
    """
    try:
        cpus = get_cpus_for_numa_node(numa_node)
        if not cpus:
            print(f"No CPUs found for NUMA node {numa_node}.")
            return

        print(f"CPUs belonging to NUMA node {numa_node}:")
        print(f"Core IDs: {cpus}")
        print(f"Total cores: {len(cpus)}")
        
        # Display current CPU usage for each core
        import psutil
        for cpu in cpus:
            usage = psutil.cpu_percent(interval=1, percpu=True)[cpu]
            print(f"Core {cpu} usage: {usage:.1f}%")
        
    except Exception as e:
        print(f"Error retrieving NUMA information: {e}")

