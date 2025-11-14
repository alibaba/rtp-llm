
# load th_transformer.so
from .ops import *
from rtp_llm.utils.torch_patch import *

# Import internal models to register them
try:
    import internal_source.rtp_llm.models_py
except ImportError:
    logging.warning("Failed to import internal_source models")
