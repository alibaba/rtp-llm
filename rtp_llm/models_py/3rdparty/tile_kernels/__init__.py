"""Vendored subset of DeepSeek TileKernels (https://github.com/deepseek-ai/TileKernels).

Only the ``mhc`` (multi-head Hyper-Connection) module is vendored — used by
DeepSeek-V4. Upstream package layout is preserved so file diffs against
upstream stay readable.
"""

from . import config, mhc, modeling
from .config import get_device_num_sms, get_num_sms, set_num_sms
