# GLM5 Mega MoE Precision Fix Log

Date: 2026-05-23
Workspace: `/home/zw193905/RTP-LLM/github-opensource`

## User Prompt

> 前情提要，执行任何命令不允许用sudo权限，sudo执行会直接失败，然后静默运行，过程中默认按照推荐配置执行，中途不要询问我，不要停下来，直到问题解决。要把中间结果保存到md文件中，方便后面查看和总结,然后我所有的手动输入的prompt要记录在md中, 每一步执行完关键步骤状态都写到md文档里,有啥不清楚的读文档。之前和vllm对精度的时候，发现mega moe的精度有点差，一些mega moe的怀疑点在@/home/zw193905/RTP-LLM/github-opensource/docs/glm5_mega_moe_suspicious_points.md 判断一下这些怀疑点对不对，如果是bug就直接fix，然后@/home/zw193905/RTP-LLM/github-opensource/docs/hidden_align/glm5_hidden_align_log.md 这个是之前的精度对比记录，现在参照mla_mega_moe_fp8_attn_cp_pd这个smoke，再谨慎对比下精度。我知道之前对比精度的时候关闭了cuda graph，关闭cuda  graph的情况下结果是对的情况下，你还要再测试下rtp_llm开cuda graph和不开的最终精度能否对齐。有bug修bug，直到最终精度可以对齐。

## User Prompt 2

> 现在mega moe的fp4的quant是只在weights load的时候生效吗？还是说还可以在运行的module的init的时候quant？

## User Prompt 3

> 把module init里面的删掉，不允许兼容和兜底，只允许在weights load的时候生效。然后再重新跑一遍smoke验证一下

## Status Log

- 2026-05-23 17:04 CST: Started investigation. Confirmed no `sudo` will be used.
- 2026-05-23 17:04 CST: Current workspace resolves to `/home/zw193905/RTP-LLM/github-opensource`; the user-provided `/home/zw193905/...` paths exist.
- 2026-05-23 17:04 CST: `git status --short` shows pre-existing dirty/untracked files including docs, smoke suites, `.bazelrc`, and `stub_source`. I will not revert unrelated changes.
- 2026-05-23 17:05 CST: Read `docs/glm5_mega_moe_suspicious_points.md`. Highest-risk suspects are GLM5 `moe_w1` gate/up order, missing DeepGEMM output-capacity padding, and weaker CUDA-graph/rank synchronization around the Mega MoE collective.
- 2026-05-23 17:05 CST: Read prior hidden-align log. Previous RTP-LLM vs vLLM hidden comparison used TP=1/EP=1 single-process dumps, disabled/eager-style execution, and concluded FP4 Mega MoE was the biggest divergence source.
- 2026-05-23 17:05 CST: `rg` is not installed in this environment; continuing with `find`/`grep`.
- 2026-05-23 17:08 CST: Located `mla_mega_moe_fp8_attn_cp_pd` in both `rtp_llm/test/smoke/suites_h20_oss.bzl` and `internal_source/rtp_llm/test/smoke/BUILD`. The smoke uses `FP8_PER_BLOCK_NO_MOE`, `MOE_STRATEGY=mega_moe`, prefill `tp=2/ep=2`, decode `dp=2/ep=2`, and decode `--enable_cuda_graph 1`.
- 2026-05-23 17:12 CST: Confirmed suspect 1 is a real Mega-wrapper bug. Common RTP `W.moe_w1` layout is `[up/value | gate]` because `silu_and_mul` reads first half as value and second half as gate. GLM5 Mega wrapper currently passes first half as gate and second half as up, while DeepGEMM Mega follows DSV4 `[gate | up]`.
- 2026-05-23 17:12 CST: Confirmed suspect 2 is a real robustness bug. DSV4 allocates output rows with `max(requested_capacity, buf.num_max_tokens_per_rank)` because DeepGEMM internally aligns the buffer capacity; GLM5 currently allocates only the requested capacity.
- 2026-05-23 17:12 CST: Suspect 5 is relevant to the requested cuda-graph check. DSV4 synchronizes CUDA-graph warmup ranks before the peer-symmetric Mega kernel; GLM5 currently does not.
- 2026-05-23 17:19 CST: Patched `rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py` so RTP stacked `[up | gate]` weights are split correctly and reordered to DeepGEMM `[gate | up]` for BF16, FP8, and load-time FP4 paths.
- 2026-05-23 17:19 CST: Patched `rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py` with DSV4-style output-capacity sizing, an explicit output-buffer bounds check, optional `GLM5_MEGA_MOE_PRE_KERNEL_BARRIER`, and CUDA-graph warmup rank synchronization before `deep_gemm.fp8_fp4_mega_moe`.
- 2026-05-23 17:19 CST: Added `rtp_llm/models_py/modules/glm5_mega_moe/test_fused_moe_wrapper_layout.py` to validate the GLM5 Mega wrapper layout conversion without requiring GPU/DeepGEMM.
- 2026-05-23 17:21 CST: Validation passed: `python -m py_compile` for the modified GLM5 Mega files succeeded.
- 2026-05-23 17:21 CST: Validation passed: `python -m unittest rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout` ran 2 tests successfully. Import printed optional `flash_attn_interface` warnings only.
- 2026-05-23 17:23 CST: `python -m black` is not installed in `/opt/conda310`; manually wrapped long lines and reran compile/unit validation successfully.
- 2026-05-23 17:27 CST: Made `docs/hidden_align/rtp_llm_dump_hidden.py` configurable for `--enable-cuda-graph`, GPU, port, dump base, and output dir so eager vs CUDA graph runs can be kept separate.
- 2026-05-23 17:27 CST: Made `docs/hidden_align/compare_hidden.py` accept `--ref-dir`, `--test-dir`, and `--out`, enabling vLLM-vs-RTP and RTP-eager-vs-RTP-cudagraph comparisons with the same metric code.
- 2026-05-23 17:27 CST: Validation passed: `python -m py_compile docs/hidden_align/rtp_llm_dump_hidden.py docs/hidden_align/compare_hidden.py`.
- 2026-05-23 17:31 CST: Ran RTP-LLM eager (`--enable_cuda_graph 0`) short prompt after the fix. Server started in 37.5s, generated 20 tokens, and saved `docs/hidden_align/rtp_llm_dumps_eager_fix/short.pt`.
- 2026-05-23 17:31 CST: Compared short prompt against existing vLLM BF16 dump. `layer03_hidden` improved to cosine=0.997267, RelL2=0.088626, MaxDiff=0.0051; prior log had cosine=0.982046, RelL2=0.376505, MaxDiff=0.0139. `final_norm` improved to cosine=0.999634, RelL2=0.027966.
- 2026-05-23 17:34 CST: Ran RTP-LLM eager medium and long_4k after the fix. Saved `docs/hidden_align/rtp_llm_dumps_eager_fix/medium.pt` and `long_4k.pt`.
- 2026-05-23 17:34 CST: Full eager-vs-vLLM comparison saved to `docs/hidden_align/comparison_eager_fix.json`. Key fixed MoE metrics: short `layer03_hidden` cosine=0.997267 RelL2=0.088626; medium cosine=0.997448 RelL2=0.071245; long_4k cosine=0.996748 RelL2=0.078184. `final_norm` RelL2 is ~0.0267-0.0280 across all prompts.
- 2026-05-23 17:35 CST: Tried RTP-LLM `--enable_cuda_graph 1` short prompt with MOEDBG hidden dump enabled. It wrote only 3 decode step dumps and then hung; this indicates MOEDBG tensor cloning/dumping is not compatible with this cuda graph verification path. Terminated the stuck debug script and child server with normal user `kill`, no `sudo`.
- 2026-05-23 17:36 CST: Added `--skip-hidden-dump` and `--query-timeout` to `docs/hidden_align/rtp_llm_dump_hidden.py`; with this flag the script saves final token outputs only and sets `MOEDBG=0`, suitable for cuda graph on/off final-precision comparison.
- 2026-05-23 17:36 CST: Validation passed: `python -m py_compile docs/hidden_align/rtp_llm_dump_hidden.py`.
- 2026-05-23 17:38 CST: Ran RTP-LLM cuda graph (`--enable_cuda_graph 1`, `MOEDBG=0`) for short/medium/long_4k and saved outputs under `docs/hidden_align/rtp_llm_outputs_cudagraph_fix/`.
- 2026-05-23 17:38 CST: Compared RTP-LLM eager vs cuda graph final output tokens. Results saved to `docs/hidden_align/comparison_eager_vs_cudagraph_fix.json`: short 20/20, medium 100/100, long_4k 50/50 exact token match.
- 2026-05-23 17:42 CST: Ran actual smoke target `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mega_moe_fp8_attn_cp_pd --config=cuda13 --test_output=streamed`.
- 2026-05-23 17:42 CST: Smoke result: PASSED in 189.3s. Runner used L20D GPUs 4-7, prefill `tp=2/ep=2`, decode `dp=2/ep=2`, decode `--enable_cuda_graph 1`. First POST saw one `8307_CACHE_STORE_LOAD_BUFFER_TIMEOUT`, runner retried once, final raw info was `ret:[True]`, `compare diff count:[0]`.
- 2026-05-23 17:43 CST: New request: remove module-init FP4 quant compatibility/fallback paths and allow GLM5 Mega MoE FP4 quantization only during weight loading. Will rerun the target smoke after the code change.
- 2026-05-23 17:44 CST: Removed the FP8 and BF16 runtime/module-init quantization branches from `MegaMoeFusedWrapper.__init__`. The wrapper now requires `moe_w1`, `moe_w2`, `moe_s1`, and `moe_s2` from load-time FP4 conversion, and requires `moe_w1/moe_w2` to be `torch.int8`.
- 2026-05-23 17:44 CST: Updated `test_fused_moe_wrapper_layout.py` so BF16 and FP8 stacked MoE weights are expected to raise, while load-time FP4 still verifies `[up | gate]` to `[gate | up]` reordering.
- 2026-05-23 17:45 CST: First rerun of the updated layout unit test failed only because the BF16 rejection assertion expected the narrower text `load-time FP4 int8`, while the actual missing-scale error says `load-time FP4 MoE weights`. Adjusted the assertion to match `load-time FP4`.
- 2026-05-23 17:45 CST: Validation passed: `python -m py_compile rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py rtp_llm/models_py/modules/glm5_mega_moe/test_fused_moe_wrapper_layout.py`.
- 2026-05-23 17:45 CST: Validation passed: `python -m unittest rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout` ran 3 tests successfully.
- 2026-05-23 18:02 CST: Re-ran `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mega_moe_fp8_attn_cp_pd --config=cuda13 --test_output=streamed` after removing module-init quant fallback.
- 2026-05-23 18:02 CST: Smoke result: PASSED in 201.7s. Final raw info was `ret:[True]`, `compare diff count:[0]`. As in the previous run, the first POST hit one `8307_CACHE_STORE_LOAD_BUFFER_TIMEOUT`, then the runner retried successfully.

## Suspicious Points Judgement

1. `moe_w1` gate/up order: confirmed bug for GLM5 Mega wrapper. RTP generic MoE layout is `[up/value | gate]`; DeepGEMM Mega expects `[gate | up]`. Fixed in `fused_moe_wrapper.py` for BF16, FP8, and load-time FP4.
2. DeepGEMM output capacity: confirmed robustness bug. Fixed in `mega_moe.py` with DSV4-style `_mega_output_capacity()` and an explicit output-buffer size check.
3. `mega_moe_enabled()` not used: partially true, but not changed in this pass. The TP=1 debug path works with world_size=1 while `mega_moe_enabled()` currently rejects world_size<=1, so blindly gating on it would break the hidden-align workflow. This is a usability/capability-gate cleanup, not the observed precision bug.
4. Token budget / CP / chunked MoE handling: plausible robustness concern, not the observed precision bug. The target smoke passed with its CP/PD config after the layout/capacity/sync fixes.
5. Missing collective/cuda-graph synchronization: confirmed relevant. Fixed by adding optional pre-kernel barrier support and CUDA-graph warmup rank synchronization before `fp8_fp4_mega_moe`.
6. `swiglu_limit`: not applicable to the tested GLM-5 4-layer config; config has no `swiglu_limit` field.
7. Packer validation: real diagnostics gap, not the observed precision bug. Left unchanged.
8. Test coverage: confirmed gap. Added a CPU unit test for wrapper layout conversion.

## Precision Summary

Eager RTP-LLM vs vLLM BF16 hidden-state comparison after the fix:

| Prompt | `layer03_hidden` Cos(row) | `layer03_hidden` RelL2 | `final_norm` Cos(row) | `final_norm` RelL2 |
| --- | ---: | ---: | ---: | ---: |
| short | 0.997267 | 0.088626 | 0.999634 | 0.027966 |
| medium | 0.997448 | 0.071245 | 0.999641 | 0.026927 |
| long_4k | 0.996748 | 0.078184 | 0.999641 | 0.026658 |

RTP-LLM eager vs RTP-LLM cuda graph final token comparison:

| Prompt | Match |
| --- | ---: |
| short | 20/20 |
| medium | 100/100 |
| long_4k | 50/50 |

## Verification Commands

- `python -m py_compile rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py rtp_llm/models_py/modules/glm5_mega_moe/test_fused_moe_wrapper_layout.py`: passed.
- `python -m unittest rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout`: passed, 2 tests.
- After removing module-init quant fallback, `python -m unittest rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout`: passed, 3 tests.
- `python docs/hidden_align/rtp_llm_dump_hidden.py --prompts short medium long_4k --enable-cuda-graph 0 ...`: passed, hidden dumps saved under `docs/hidden_align/rtp_llm_dumps_eager_fix/`.
- `python docs/hidden_align/compare_hidden.py --ref-dir docs/hidden_align/vllm_dumps --test-dir docs/hidden_align/rtp_llm_dumps_eager_fix --out docs/hidden_align/comparison_eager_fix.json`: passed.
- `python docs/hidden_align/rtp_llm_dump_hidden.py --prompts short medium long_4k --enable-cuda-graph 1 --skip-hidden-dump ...`: passed, final outputs saved under `docs/hidden_align/rtp_llm_outputs_cudagraph_fix/`.
- `python docs/hidden_align/compare_hidden.py --ref-dir docs/hidden_align/rtp_llm_dumps_eager_fix --test-dir docs/hidden_align/rtp_llm_outputs_cudagraph_fix --out docs/hidden_align/comparison_eager_vs_cudagraph_fix.json`: passed, exact token match.
- `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mega_moe_fp8_attn_cp_pd --config=cuda13 --test_output=streamed`: passed.
- After removing module-init quant fallback, `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mega_moe_fp8_attn_cp_pd --config=cuda13 --test_output=streamed`: passed.
- `git diff --check` on modified investigation/code files: passed.
- After removing module-init quant fallback, `git diff --check` on the touched files: passed.

## Final Workspace Notes

- No lingering `rtp_llm.start_server`, `rtp_llm_backend_server`, dump, or bazel test processes were found after verification.
- All GPUs returned to idle-level memory usage (~1 MiB shown by `nvidia-smi`).
- After the final smoke rerun, no lingering RTP-LLM/Bazel test processes were found and all GPUs showed 0 MiB used by `nvidia-smi`.
- Pre-existing dirty files such as `.bazelrc` and `stub_source` were not modified by this investigation.
