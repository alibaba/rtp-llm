"""Set preferred host-memory node before exec; both schemes use the same policy."""

import ctypes
import os
import sys

node = int(os.environ.get("DSV4_NUMA_NODE", "1"))
if node >= 0:
    numa = ctypes.CDLL("libnuma.so.1")
    if numa.numa_available() < 0 or node > numa.numa_max_node():
        raise RuntimeError(f"NUMA node {node} is unavailable")
    numa.numa_set_preferred(node)
    if numa.numa_preferred() != node:
        raise RuntimeError(f"could not select NUMA node {node}")
os.execvp(sys.argv[1], sys.argv[1:])
