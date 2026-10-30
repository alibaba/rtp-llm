"""Startup configuration for the validated GLM5 prefill optimization bundle."""

import os

# Configure before importing model modules; unsupported layouts keep their fallback.
ENABLED = os.environ.get("RTP_GLM5_PREFILL_OPTIMIZE", "0") == "1"
