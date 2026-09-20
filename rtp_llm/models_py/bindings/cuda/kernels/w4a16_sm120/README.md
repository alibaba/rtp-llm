# SM120 W4A16 FFN

Enable the supported dense Qwen target FFN path before loading the model:

```bash
ENABLE_W4A16_SM120_DENSE_FFN=1
# Or pass the CLI option:
--enable_w4a16_sm120_dense_ffn true
```

The default is off. CLI values take precedence over the environment variable.
The quantization argument group binds the value to the target ModelConfig;
draft models remain disabled. Restart the service to change this setting.
There are no changes to checkpoint formats or the Linear factory.

The model retains its BF16 weights and prepares one rotated INT4/group8 copy
at initialization. Activations use normalized block-128 Hadamard transforms
with fixed signs. Both FFN projections switch together when the physical
input has `0 < M < 64` rows. CUDA Graph uses the captured physical row count,
including padding and speculative verification tokens, not KV history length.
At `M >= 64`, the original BF16 projections run without rotation.

MoE, other weight quantization, LoRA, FFN disaggregation, non-BF16 precision,
and non-SM120 devices are rejected at startup. Online weight updates are
disabled for model instances loaded with this feature. Account for another
0.625 bytes per accelerated FFN weight when sizing KV cache memory.

## Unmodified Source and Deferred Issue

The ten `.cu`/`.cuh` files originate from `w4a16_gemm_sm120.tar.gz`, SHA256
`5a1493b98ef837a3db040cf5f0a5b3f456e28db149aad0e5333145d93f33a483`.
Their algorithms, quantization, rounding, automatic split-K and BF16 atomic
reduction are unchanged. Source redistribution authorization must be checked
before publishing these supplied files outside this workspace.

The supplied `load_W_tile`/`load_S_tile` do not mask partial N/K tiles, although
the original C entry point permits them. This remains a deferred upstream
issue, not a fix in this integration. The tensor binding rejects unsafe
shapes; FFN initialization conservatively requires N divisible by 256 and K
divisible by 128 for both projections, otherwise that FFN remains BF16.
Enabling the feature when no FFN qualifies is a startup error.

## Dependencies

Build RTP compute ops with the CUDA13 x86 or CUDA12.9 configuration. The
dedicated target compiles the supplied kernel for SM120a only. Python imports
the built `rtp_llm_ops` bindings; it never loads the supplied prebuilt library.

FHT is fetched at a pinned upstream commit and checksum by
`//3rdparty/fast_hadamard_transform:repositories.bzl`, compiled by Bazel against
RTP's `torch_deps()`, and statically linked into `librtp_compute_ops.so`.
The block128 wrapper uses `rtp_llm_ops.w4a16_sm120_hadamard`; no external FHT
wheel, local build, virtualenv, or runtime compilation is required.
The RTP wheel also contains the upstream FHT license. Standard image packaging
installs this complete RTP wheel and its declared Python dependencies.
