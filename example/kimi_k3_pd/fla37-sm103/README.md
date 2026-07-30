# Kimi K3 decode-accuracy recurrent KDA image

This directory contains the validated FLA 0.5.1 / Triton 3.7 recurrent KDA
kernel image used by `KIMI_K3_EXECUTION_MODE=accuracy` on SM103.

- Kernel: `fused_recurrent_kda_fwd_kernel.cubin`
- SHA256: `6a486a94bae75c683e1ef961e2bfa9cdb18270f1c6980a8d3097f766164f3374`
- Target: CUDA SM103
- Triton compiler version: 3.7.0

The runtime verifies the SHA256, architecture, Triton metadata, launch shape,
and K3 Decode tensor shapes before replacing the local Triton 3.6 image.
Optimized Prefill does not use this image.
