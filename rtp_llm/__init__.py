import time
st = time.time()
# load th_transformer.so
# Import internal models to register them
from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.torch_patch import *
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

enable_compile_monitor()

from .ops import *

if has_internal_source():
    import internal_source.rtp_llm.models_py

consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
