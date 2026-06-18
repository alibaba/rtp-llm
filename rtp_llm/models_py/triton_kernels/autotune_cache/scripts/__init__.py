# Triton autotune cache generation/extraction tools.
#
# These scripts populate
# rtp_llm/models_py/triton_kernels/autotune_cache/configs/{GPU}/*.json by:
#   1. Running representative kernels (one driver per op family under
#      generators/) under triton.autotune to fill the Triton on-disk cache
#      (~/.triton/cache/.../*.autotune.json).
#   2. Reading those raw cache files and writing per-kernel best-config
#      JSONs that ship with the wheel.
#
# Usage:
#   python -m rtp_llm.models_py.triton_kernels.autotune_cache.scripts.extract \
#       --init-default --op kda
