import time

from rtp_llm.utils.jit_cache_manager import apply_jit_cache_env_from_env

st = time.time()

apply_jit_cache_env_from_env()
try:
    import triton
except ImportError:
    pass

# load th_transformer.so
# Import internal models to register them
from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.torch_patch import *
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

from .ops import *

# check triton version
# if triton.__version__ < "3.4":
#     enable_compile_monitor()


# enable_compile_monitor()


if has_internal_source():
    import internal_source.rtp_llm.models_py


consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
