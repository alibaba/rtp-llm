cd ../../bazel-bin

PYTHONPATH=. LD_LIBRARY_PATH=./rtp_llm/libs:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/extras/CUPTI/lib64/ ~/.local/bin/pybind11-stubgen libth_transformer_config

# NOTE: core, but can generate pyi
PYTHONPATH=. LD_LIBRARY_PATH=./rtp_llm/libs:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/extras/CUPTI/lib64/ ~/.local/bin/pybind11-stubgen libth_transformer

# NOTE: core now
# PYTHONPATH=. LD_LIBRARY_PATH=./rtp_llm/libs:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/extras/CUPTI/lib64/ ~/.local/bin/pybind11-stubgen librtp_compute_ops
