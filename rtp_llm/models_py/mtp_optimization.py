"""Startup switch for the MTP fusion bundle.

Setting RTP_LLM_MTP_OPTIMIZE=0 selects the existing unfused MTP paths.
The GLM5 prefill switch is independent of this decode-only switch.
"""

import os

ENABLED = os.environ.get("RTP_LLM_MTP_OPTIMIZE", "0") == "1"
