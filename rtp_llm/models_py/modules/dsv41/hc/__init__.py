"""DeepSeek-V4 Hyper-Connection implementations."""

from rtp_llm.models_py.modules.dsv41.hc.base import HCMode
from rtp_llm.models_py.modules.dsv41.hc.factory import build_hc_head, build_hc_unit

__all__ = ["HCMode", "build_hc_head", "build_hc_unit"]
