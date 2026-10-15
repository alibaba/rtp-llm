# SM120 W4A16 FFN

Optional INT4/group8 weight quantization with BF16 activations for supported
Qwen dense FFNs on NVIDIA SM120 GPUs. Weights use block-128 Hadamard rotation
and are prepared from BF16 source weights during loading.

## Usage

Enable before starting the service (disabled by default):

```bash
export ENABLE_W4A16_SM120_DENSE_FFN=1
```

Or add the CLI option to the server command:

```bash
--enable_w4a16_sm120_dense_ffn true
```

CLI values take precedence over the environment variable. Restart the service
to change this setting; draft models remain disabled.

## Requirements and Behavior

- Requires the Python model path, BF16 compute and unquantized BF16 source
  weights, plus RTP compute ops built with CUDA 12.9 x86 or CUDA 13 x86.
- Supports online FP8 quantization alongside W4A16; pre-quantized checkpoints,
  MoE, LoRA and FFN disaggregation are not supported. Online weight updates
  are disabled.
- Both FFN projections use W4A16 when the physical input row count is
  `0 < M < 64` (including CUDA Graph padding). Otherwise they use the default
  BF16 or online-FP8 path.
- Each local weight matrix `[N, K]` requires `N % 256 == 0` and `K % 128 == 0`.
  Unsupported FFNs use the default path; startup fails if no FFN qualifies.
- The additional packed weights and scales cost 0.625 bytes per accelerated
  FFN weight, on top of the default-path weights.
