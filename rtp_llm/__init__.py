import time

st = time.time()
import triton
import torch
# load th_transformer.so
# Import internal models to register them
from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.torch_patch import *
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

# check triton version
if triton.__version__ < "3.5" and tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
    enable_compile_monitor()

from .ops import *

if has_internal_source():
    import internal_source.rtp_llm.models_py

consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
