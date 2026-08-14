# vLLM FlashAttention CuTeDSL

## Source Pin

- Repository: https://github.com/vllm-project/flash-attention
- Commit: `ed4b7342bc8f0489dd9b649d5288867e35fc6a32`
- Selected by vLLM commit: `6e311c6e2014a54868e2cbedf944a6873d06228b`
- Archive SHA-256: `fb0107dc613515ddf707a275ba7dbcab86f24853dc01608cfd0dcf4a33664018`

`upstream/flash_attn/cute` is an unmodified copy of the directory at that
commit. `SHA256SUMS` records every file in the upstream snapshot.

## Bazel Integration

The Bazel target generates a standalone `vllm_flash_attention.cute` package
from the upstream Python sources. It mechanically replaces `flash_attn.cute`
imports in generated files, avoiding a collision with RTP-LLM's installed
FlashAttention 2 package. The generated package remains physically rooted in
this third-party directory; the RTP-LLM wheel strips that repository prefix
when packaging it.

The runtime package is exposed as:

```text
//3rdparty/vllm_flash_attention:fa4_cute
```

RTP-LLM code imports it with:

```python
from vllm_flash_attention.cute.interface import _flash_attn_fwd
```

## Runtime Dependencies

The CUDA 12.9 x86 lock currently resolves the FA4-specific dependencies to:

| Package | Version | Purpose |
| --- | --- | --- |
| `apache-tvm-ffi` | `0.1.13` | CuTeDSL runtime and compiled-function ABI |
| `nvidia-cutlass-dsl` | `4.5.3` | CuTeDSL compiler and runtime |
| `quack-kernels` | `0.5.0` | Shared CuTeDSL kernel helpers |
| `torch-c-dlpack-ext` | `0.1.5` | Torch/DLPack interop used by FA4 |

FA4 also uses the existing RTP-LLM `torch` and `einops` dependencies.

The vendored upstream `pyproject.toml` requests CUTLASS DSL 4.6 and QuACK
0.5.3 or newer. Those versions are not used because CUTLASS DSL 4.6 requires
`protobuf>=6.30.2`, while RTP-LLM currently pins `protobuf==4.25` and
`grpcio-tools==1.57.0`. QuACK 0.5.3 also selects the CUTLASS DSL 4.6 line.
Upgrading to those versions therefore requires a project-wide protobuf/gRPC
dependency migration, not a local FA4 version bump.

CUTLASS DSL 4.5.3 requires an explicit result type for one `nvvm.fmax` call in
the pinned upstream `utils.py`. The Bazel generation rule inserts `T.f32()`
into the generated copy and first verifies that exactly one expected callsite
exists. The upstream snapshot remains unchanged.

## Upgrade Procedure

1. Select the vLLM commit to follow and record the exact
   `vllm-project/flash-attention` commit pinned by that vLLM revision. Do not
   follow a moving branch or tag.
2. Download an archive for that exact FlashAttention commit, record the archive
   SHA-256 above, and replace the complete `flash_attn/cute` directory under
   `upstream/`. Do not edit the copied files.
3. Compare the copied directory against the archive and inspect the upstream
   `LICENSE`, `pyproject.toml`, Python package layout, and `_flash_attn_fwd`
   interface before making RTP-LLM changes.
4. Update `_CUTE_PY` in `BUILD` so every upstream Python file has a declared
   Bazel output. The generation rule fails the build when the upstream Python
   file count no longer matches `_CUTE_PY`. Keep documentation and packaging
   files in the snapshot; besides the Python sources, only `LICENSE` and
   `AUTHORS` enter the runtime target and the released wheel.
5. Re-evaluate every upstream dependency against RTP-LLM's CUDA 12.9 lock.
   Update `deps/requirements_torch_gpu_cuda12_9.txt`, regenerate
   `deps/requirements_lock_torch_gpu_cuda12_9.txt`, and keep the wheel metadata
   in `rtp_llm/BUILD` consistent with the resolved versions.
6. Re-evaluate the namespace replacement and the CUTLASS DSL 4.5.3 `nvvm.fmax`
   adaptation. Remove or change an adaptation only after confirming the API of
   the selected dependency versions. The guarded callsite count should fail the
   build when upstream source structure changes unexpectedly.
7. Update the commit metadata in this README and
   `vllm_flash_attention/__init__.py`, then regenerate the manifest from this
   directory:

   ```bash
   find upstream/flash_attn/cute -type f -print0 \
     | sort -z \
     | xargs -0 sha256sum > SHA256SUMS
   sha256sum -c SHA256SUMS
   ```

8. Verify that generated sources contain no remaining `flash_attn.cute`
   imports, build the RTP-LLM wheel, and inspect the wheel for
   `vllm_flash_attention/cute/*.py` plus `vllm_flash_attention/LICENSE` and
   `vllm_flash_attention/AUTHORS`, all without the repository path prefix.
9. Run the FA4 MTP unit test on SM90/CUDA 12.9:

   ```text
   //rtp_llm/models_py/modules/factory/attention/cuda_impl/test:test_py_flash_attn_v4_mtp
   ```

10. Confirm target-verify, draft-forward, CUDA graph replay, and numerical
    reference coverage before updating the source pin.
