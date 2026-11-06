# ARM SM100 / Torch 2.13 native build

The fresh native build used ARM64 Python 3.10.9, Torch 2.13.0+cu130 with CXX11 ABI,
CUDA toolkit 13.2, GCC 10, Bazel 6.4.0 and installed NCCL 2.29.7. No old SM103 or
Torch 2.11 binary was reused. The accompanying script takes a checkout of this
branch that includes the Torch 2.13 `SpeculativeSampler` `torch::rand` overload
fix and the world72 initial-pause implementation. It does not fetch models,
install Python packages, deploy artifacts or start GPU work.

```bash
python build_torch213_sm100.py \
  --source /path/to/rtp-llm \
  --output /path/to/fresh-build-output \
  --bazelisk /path/to/bazelisk --jobs 32 --ram-mb 192000
```

Use the selected interpreter directly; resolving a virtualenv interpreter
symlink before execution can silently select the base environment. Run inside
the authorized development container and supervise the build with a real parent
`wait`, with scoped PID and start-time identity checks for any cancellation.

The external repository label still contains `torch_2.11` because it is the
branch's Bazel dependency name. `--override_repository` points that label to the
**installed Torch 2.13** headers and libraries. Header search uses Bazel
`includes` (SYSTEM) rather than `strip_include_prefix` for both public and C++ API
headers. The generated include tree omits Torch's vendored `google` directory;
otherwise its protobuf headers shadow the separate protobuf version used by
RTP. A C++17 `-Werror -fsyntax-only` probe checks that setup before compilation.
It does not globally lower warnings or change RTP's C++ standard.

The exact effective target selection is `--config=cuda13_arm` and
`TF_CUDA_COMPUTE_CAPABILITIES=10.0` in action, host-action and repository env.
Installed NVSHMEM, cuSPARSELt, NCCL and cuDNN SDKs are supplied through explicit
external repository overrides. NCCL include/install and cuDNN install paths are
also passed to actions. The complete resolved command, source revision, changed
tracked-file hashes, interpreter ABI and SDK paths are written to `BUILD-INPUT.json`.
The four targets are `//:th_transformer`, `//:th_transformer_config`,
`//:th_grammar_tokenizer_info` and `//:rtp_compute_ops`.

The observed fresh build completed 14,074 actions in **824.846 seconds** with
actual build and supervisor wait return codes 0. These timings describe that
one build, not a performance promise. The first build also required the
Torch 2.13 `torch::rand` fix; a previous failed build is not its success receipt.

Deploying only those four libraries is insufficient: their `DT_NEEDED` closure
included **11 `libflashinfer_*.so`** libraries from the same fresh Bazel outputs.
Bundle those exact libraries as well, hash every artifact, and keep existing
system SDK dependencies explicit. Do not copy a build cache or rely on Bazel
RPATHs in production. The accepted deployment imported through
`rtp_llm.ops.ensure_engine_ops_loaded()` in the public config/compute/grammar/
transformer order. Its world72 initial-pause marker was 1. Independent imports
ran from a source-only runtime landing with no build/Bazel path on `PYTHONPATH`
or `LD_LIBRARY_PATH`; actual mapped-library paths, inodes and hashes were checked.

Python FlashInfer and `rtp_kernel` are separate runtime dependencies from these
native `libflashinfer_*.so` files. Pin them independently, audit any auxiliary
version changes, and test the production-selected operators against the actual
SM100/Torch runtime before loading a multi-terabyte model.

A successful build, import, descriptor-on-meta check, communication toy or tiny
operator test is **not full-model RL acceptance**. Full acceptance still needs
complete streamed weight coverage, paused-to-resume control, actual rollout and
training updates, weight synchronization and cleanup on the full allocation.
