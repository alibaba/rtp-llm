# SPDX-License-Identifier: Apache-2.0
"""
Site-specific configuration for Python runtime.

This file is automatically imported by Python at startup (before any user code).
It fixes the nvidia-cutlass-dsl import path issue.
"""

import sys
import os

# Fix nvidia-cutlass-dsl cutlass module path
try:
    import nvidia_cutlass_dsl
    pkg_dir = os.path.dirname(nvidia_cutlass_dsl.__file__)
    python_packages_dir = os.path.join(pkg_dir, 'python_packages')
    
    if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
        sys.path.insert(0, python_packages_dir)
except ImportError:
    pass  # nvidia-cutlass-dsl not installed
