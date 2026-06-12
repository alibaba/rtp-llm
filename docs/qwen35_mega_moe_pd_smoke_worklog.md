# Qwen35 MegaMoE PD Smoke Worklog

Last updated: 2026-06-08 CST

This file is the compact handoff log. Repeated polling logs, long GPU snapshots,
and duplicated failure traces were removed. User prompts are kept below, deduped
where the same prompt was recorded multiple times.

## User Prompts

1. `前情提要，执行任何命令不允许用sudo权限，sudo执行会直接失败，然后静默运行，过程中默认按照推荐配置执行，中途不要询问我，不要停下来，直到问题解决。要把中间结果保存到md文件中，方便后面查看和总结,然后我所有的手动输入的prompt要记录在md中, 每一步执行完关键步骤状态都写到md文档里,有啥不清楚的读文档。@/home/zw193905/RTP-LLM/internal_source/rtp_llm/test/smoke/BUILD 中有glm5模型的smoke测试mla_mtp_mega_moe_cudagraph_pd_full_ckpt，我现在要你加一个qwen35的smoke测试，可以参照前面glm5的，其中qwen35的模型文件在/home/zw193905/models/Qwen3.5-397B-A17B-FP8。这里面你需要支持一下qwen35的mega moe和prefill cp。然后只用后面4张卡，prefill和decode各用两张卡，moe fp4的情况下应该放得下。smoke测试命令可以参考bazelisk test //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt --test_timeout=3000 --config=cuda13。在跑smoke测试之前可以先看看卡是不是空闲的。@/home/zw193905/RTP-LLM/github-opensource/docs/qwen35_mega_moe_pd_smoke_worklog.md 这个是之前跑了一半的任务记录，接着跑吧`
2. `跑smoke的时候可以等后面4张卡都空闲了再跑`
3. `不允许改gpu设备状态`
4. `qwen35 mtp的attention实现可以参考6e45b58d69c045343317169e9b324b2dee80e291这个commit，可以cherry pick这个commit的实现`
5. `首先我将你生成的smoke测试中的参数稍微改了一下，现在不测试cp了，然后prefill和decode都改成了4卡，你现在可以跑所有的卡了。smoke的stress test参数你得给我加上啊，测试并发下的正确性。然后重新跑这个测试，fix bug，直到跑通。fix bug过程中不要随意的改上面的一些功能和参数。改功能和参数一定要谨慎。等这个smoke测试跑通之后，再继续添加一个不开mtp的smoke测试，参数上应该只把mtp相关的参数关掉，不应该动其他的配置。smoke测试都添加完之后。这两个smoke都run一次确保都成果，修改互相不冲突。在@/home/zw193905/RTP-LLM/internal_source/rtp_llm/test/perf_test/BUILD 中添加这个smoke相关的perf test，可以参考glm5_mega_moe_8dp8ep_grid_mtp_test和glm5_mega_moe_8dp8ep_grid_prefill_mtp_test上面的一些测试参数，例如batch size和seq len之类的测试参数，但是功能参数还是要以smoke测试上的为准。我要你添加不开mtp的情况下的prefill和decode各自4卡和8卡的perf test，以及开mtp情况下的4卡和8卡的perf test。prefill都是tp，decode都是dp。每个perf test成功跑完之后产出的perf result，例如timelines之类都copy 到@/home/zw193905/docs_scripts 目录下的相应位置。`
6. `别瞎测试了，明显mtp接受率和decode tpot都不对`
7. `iter轮数太多了，改到10就行了，没必要30`
8. `你先把不带mtp的任务先完成吧，然后再继续mtp相关的任务`
9. `先暂停任务吧，然后把中间过程记录保存好，方便后面接手继续干`
10. `md文件中没有用的信息太少了，精简一下，不过要保留user prompt`
11. `@/home/zw193905/RTP-LLM/github-opensource/docs/qwen35_mega_moe_pd_smoke_worklog.md 接着这个任务记录继续干吧，然后还是先把非mtp的搞完再搞mtp，有bug，结果不对就对精度。如何启动vllm作为精度对比可以参考@/home/zw193905/RTP-LLM/github-opensource/docs/hidden_align/glm5_hidden_align_log.md 作为使用指导。`
12. `现在卡空闲了，开始继续任务吧，没让你停之前别停`

## Current Priority

- Pause MTP work for now.
- Finish no-MTP smoke first.
- Keep `CONCURRENT_STRESS_ITERS=10`.
- Do not use `sudo`.
- Do not casually change topology or feature parameters.
- Before a new run, check that GPUs are idle.

## Current No-MTP Smoke Target

Target:

```bash
bazelisk test //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp \
  --test_timeout=16000 --config=cuda13 --test_output=streamed --nocache_test_results
```

Intended no-MTP topology and key params:

- Prefill: `tp_size=4`, `dp_size=1`, `ep_size=4`, `world_size=4`.
- Decode: `tp_size=1`, `dp_size=4`, `ep_size=4`, `world_size=4`.
- No MTP args.
- MegaMoE enabled with `MOE_STRATEGY=mega_moe` and `--moe_strategy mega_moe`.
- Quantization: `FP8_PER_BLOCK_NO_MOE`.
- FP8 KV cache enabled.
- Decode CUDA graph enabled with capture config `1,2,4,8,16`.
- Stress enabled on decode:
  - `CONCURRENT_STRESS_ITERS=10`
  - `CONCURRENT_STRESS_CONCURRENCY=16`
  - `CONCURRENT_STRESS_MAX_NEW_TOKENS=2048`
  - repeat detection / fail-test / halt-on-event enabled.
- Current memory mitigation: `ENABLE_FP32_LM_HEAD=0` and `--enable_fp32_lm_head 0`.

## Key Code Changes To Preserve

Qwen35 linear attention FP8 qkv/z layout fix:

- `rtp_llm/model_loader/linear_attn_weight.py`
  - Added grouped qkv/z merge helper for Qwen35 linear attention.
- `rtp_llm/models/qwen3_next/qwen3_next_weight.py`
  - Uses the grouped merge for `linear_attn.in_proj_qkvz.weight`.
- `rtp_llm/model_loader/per_block_fp8_quant_weight.py`
  - Applies the same grouped merge in FP8 per-block weight loading.
- `rtp_llm/model_loader/test/test_qwen35_linear_attn_weight.py`
  - Unit coverage for grouped qkv/z merge.
- `rtp_llm/model_loader/test/BUILD`
  - Registers the new unit test.

Earlier relevant runtime fixes that are still important:

- `rtp_llm/models_py/modules/factory/attention/common.py`
  - Added reshape helpers for paged KV cache / scale with physical block size.
- `rtp_llm/models_py/model_desc/qwen3_next.py`
  - Hybrid full-attn cache/scale reshape uses physical `seq_size_per_block`.
  - Linear attention uses its physical block map.
  - Full attention uses kernel block map.
  - CUDA graph full-attn layer metadata refresh was added.

## Verified So Far

Passed:

```bash
bazelisk test //rtp_llm/model_loader/test:test_qwen35_linear_attn_weight \
  --config=cuda13 --test_output=streamed --nocache_test_results
```

Passed:

```bash
bazelisk build //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp \
  --config=cuda13
```

Not yet passed:

- no-MTP smoke has not completed successfully yet.
- no-MTP perf tests have not been run yet.
- MTP smoke/perf is intentionally deferred.

## Latest No-MTP Run State

Before the final pause:

- A no-MTP run with `ENABLE_FP32_LM_HEAD=1` failed during startup/weight loading with CUDA OOM.
- The clearest OOM was prefill rank0:
  - tried to allocate `256.00 MiB`
  - GPU0 had only `136.94 MiB` free
- After that, only no-MTP `ENABLE_FP32_LM_HEAD` / `--enable_fp32_lm_head` was changed from `1` to `0`.
- The next no-MTP run started cleanly, reached frontend/DashSc ready, and was still loading weights.
- No `OutOfMemory`, `Traceback`, `CUDA error`, `illegal memory`, `MALLOC`, or backend died was observed before the user asked to pause.
- The run had not reached `load weights took`, query dump, or stress.
- The run was intentionally terminated with normal `kill -TERM` to pause.
- After cleanup, no `bazelisk`, `rtp_llm_rank`, `start_server`, or frontend process remained.
- Final GPU state after cleanup: all 8 GPUs were `0 MiB`, `0%`.

## Resume Log — 2026-06-08 Evening

- Resumed from this handoff with the same priority: finish no-MTP before MTP.
- Verified the no-MTP smoke target still has no MTP args, keeps
  `CONCURRENT_STRESS_ITERS=10`, and keeps the intended 4-card prefill /
  4-card decode topology.
- Verified the no-MTP smoke target still builds:

```bash
bazelisk build //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp --config=cuda13
```

- Build result: passed.
- GPU check at `2026-06-08 19:31 CST`: all 8 GPUs were busy, with about
  `129525-157445 MiB` used and up to `100%` GPU utilization. `nvidia-smi`
  did not expose running process rows. No GPU state was modified.
- Existing Bazel testlog under
  `bazel-testlogs/internal_source/rtp_llm/test/smoke/next_mega_moe_cudagraph_pd_full_ckpt_no_mtp`
  was inspected. It did not show a model/runtime error; that run used the
  default Bazel timeout and was killed at `2026-06-08 16:27:21 CST` while
  weight loading was still in progress.
- Re-ran the no-MTP smoke after GPUs became idle at `2026-06-08 19:42:34 CST`:

```bash
bazelisk test //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp \
  --test_timeout=16000 --config=cuda13 --test_output=streamed --nocache_test_results
```

- Startup succeeded. Both prefill and decode ports became ready:
  - prefill port `13984` ready at `2026-06-08 19:57:06 CST`
  - decode port `14716` ready at `2026-06-08 19:57:44 CST`
- The two normal smoke queries succeeded and produced actual dumps with no
  compare diff.
- Stress failed in iter 1 with `CONCURRENT_STRESS_CONCURRENCY=16` and
  `CONCURRENT_STRESS_MAX_NEW_TOKENS=2048`.
  - `total_requests=16`
  - `http_errors=0`
  - `repetition_events=8`
  - repeated fragments included `ER` and ` sp`
  - `HALT_ON_EVENT=1` stopped the run after the first failure event
  - `avg_accept=1.0`, as expected for no-MTP
- Failure shape: long nonsensical responses containing repeated `ER`/`sp`
  fragments under concurrent decode; baseline single queries were OK.
- After smoke cleanup at `2026-06-08 20:01 CST`, no RTP-LLM/Bazel server
  processes remained and GPUs were back to near-empty (`1-4 MiB`, `0%`).

### Correction — 2026-06-08 20:11 CST

- Re-inspected the actual dump files from the same no-MTP smoke run:
  - `q_r_qwen35_397b_fp8_mega_moe_cudagraph_pd_full_ckpt.query_0.json`
  - `q_r_qwen35_397b_fp8_mega_moe_cudagraph_pd_full_ckpt.query_1.json`
- The normal smoke query path only appeared to pass because the smoke compare
  skips content for this case. The generated content is already bad before the
  concurrent stress phase:
  - query 0 starts with `.ERERER:// sp\"ERER`
  - query 1 starts with `\"ERERER.py\"\"`
- Current conclusion: the no-MTP failure is a precision/model-path/logit or
  weight-layout bug, not only a high-concurrency stress issue.
- GPU check at `2026-06-08 20:11 CST`: all 8 GPUs were busy, with about
  `234237-243307 MiB` used and GPU util ranging from `0%` to `98%`.
- vLLM environment found for reference runs:
  - `/opt/conda310/bin/python`
  - vLLM `0.21.0`
  - Transformers `5.9.0`
  - Torch `2.11.0+cu130`
- Next debugging direction: once GPUs are idle, produce a vLLM greedy baseline
  for the two smoke prompts, then compare RTP-LLM against it. If vLLM output is
  sane and RTP output remains `ER`/`sp` garbage, prioritize hidden/logit
  alignment and Qwen35 full/linear attention weight layout checks.
- Added `docs/qwen35_vllm_no_mtp_baseline.py` for that baseline. It uses
  `/opt/conda310/bin/python`, TP=8 by default, greedy sampling
  (`temperature=0`, `top_k=1`, `max_tokens=32`), chat-template formatted
  versions of the two smoke prompts, and saves
  `docs/qwen35_vllm_no_mtp_baseline.json`.
- Syntax check passed:

```bash
/opt/conda310/bin/python -m py_compile docs/qwen35_vllm_no_mtp_baseline.py
```
- Offline tokenizer check with the same HF tokenizer/template produced the same
  prompt lengths as the RTP dump:
  - `What is the capital of France?` -> 17 tokens
  - `Translate to French: 'Hello, how are you today?'` -> 22 tokens
- This reduces the likelihood that the `ER`/`sp` output is caused by a chat
  renderer or tokenizer mismatch.
- Added env-gated `MOEDBG` tensor dumps to
  `rtp_llm/models_py/model_desc/qwen3_next.py`, mirroring the existing GLM/DSV4
  hidden-align mechanism. With `MOEDBG=0` it remains disabled; with `MOEDBG=1`
  it records `embed_out`, per-layer hidden/residual/combined, and `final_norm`.
- Syntax check passed:

```bash
/opt/conda310/bin/python -m py_compile rtp_llm/models_py/model_desc/qwen3_next.py
```
- no-MTP smoke target still builds after the debug hook:

```bash
bazelisk build //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp --config=cuda13
```
- Waited for GPUs from `2026-06-08 20:22 CST` to `20:37 CST`. They did not
  become idle; all 8 GPUs stayed at about `234-243 GiB` memory used, with
  GPUs 0-3 still showing compute activity. No GPU state was changed and no
  process was killed.
- Added `docs/qwen35_rtp_no_mtp_standalone_debug.py` as a non-PD isolation
  runner. It starts RTP-LLM with Qwen35 no-MTP, FP8_PER_BLOCK_NO_MOE,
  MegaMoE, TP=8/EP=8 by default, queries the same two chat prompts with
  `debug_info=true`, and can optionally enable `MOEDBG`.
- Syntax check passed:

```bash
/opt/conda310/bin/python -m py_compile docs/qwen35_rtp_no_mtp_standalone_debug.py
```
- Added `docs/qwen35_vllm_hidden_dump.py` for vLLM prefill hidden-state dumps
  of the same two smoke prompts. It follows the GLM hidden-align approach:
  in-process vLLM engine, hooks `embed_tokens`, every decoder layer, and final
  norm, then writes `.pt` plus `.stats.json` under
  `docs/qwen35_vllm_hidden_dumps/`.
- Syntax check passed:

```bash
/opt/conda310/bin/python -m py_compile docs/qwen35_vllm_hidden_dump.py
```
- Latest GPU check at `2026-06-08 20:39 CST` was still busy:
  `0..7` had about `234237-243307 MiB` used; GPUs 0-3 still had active
  utilization. vLLM/RTP large-model debug runs were not started.

## Resume Log — 2026-06-09

- User reported GPUs are idle and asked to continue without stopping.
- Verified at `2026-06-09 10:15 CST`: all 8 GPUs were `0 MiB`, `0%`.
- Current order remains no-MTP first, MTP deferred.
- Since prior query dumps were already bad, the immediate next step is vLLM
  baseline generation for the two smoke chat prompts, not another blind smoke
  rerun.
- Correction from user: vLLM baseline only needs 4 GPUs for this model; using
  8 GPUs was unnecessary and caused a `gpu_memory_utilization=0.9` free-memory
  threshold failure during vLLM worker startup. Future vLLM baseline runs should
  use 4 GPUs first, with a lower `--gpu-mem` only if vLLM's allocator still
  rejects startup.
- The failed vLLM startup still revealed a useful precision clue before aborting:
  vLLM resolves the model as `Qwen3_5MoeForConditionalGeneration` and logs
  `Auto-disabled DeepGemm for model_type=qwen3_5_moe_text on Blackwell.
  DeepGemm E8M0 scale format causes accuracy degradation for this architecture.
  Falling back to CUTLASS.`
- RTP currently uses DeepGEMM/UE8M0 for FP8 per-block linear weights on SM10x.
  Next local fix is to mirror the vLLM guard for Qwen3.5: keep fp32 per-block
  scales instead of converting them to DeepGEMM UE8M0 packed scales, and make
  the runtime linear path infer its layout from the actual scale dtype.

## Perf Targets

No-MTP perf targets were added earlier in `internal_source/rtp_llm/test/perf_test/BUILD`, but have not been run successfully yet:

- `qwen35_mega_moe_4dp4ep_grid_no_mtp_test`
- `qwen35_mega_moe_4tp4ep_grid_prefill_no_mtp_test`
- `qwen35_mega_moe_8dp8ep_grid_no_mtp_test`
- `qwen35_mega_moe_8tp8ep_grid_prefill_no_mtp_test`

MTP perf work is deferred until no-MTP smoke/perf is done.

## Next Steps

1. Confirm GPUs are idle:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

2. Re-run no-MTP smoke:

```bash
bazelisk test //internal_source/rtp_llm/test/smoke:next_mega_moe_cudagraph_pd_full_ckpt_no_mtp \
  --test_timeout=16000 --config=cuda13 --test_output=streamed --nocache_test_results
```

3. Watch for:

- `load weights took`
- service ready
- actual query dumps under `bazel-testlogs/.../test.outputs/smoke_actual/...`
- 10 stress iterations
- repetition / correctness failures
- OOM during final weight placement

4. If no-MTP smoke passes, run no-MTP perf targets and copy result artifacts/timelines to `/home/zw193905/docs_scripts`.

5. Resume MTP only after no-MTP smoke and no-MTP perf are handled.

## Progress — 2026-06-09 10:39 CST

- Corrected vLLM baseline script to default to 4 GPUs (`--tp 4`) and added
  `skip_mm_profiling=True` so a text-only baseline does not enter the visual
  encoder profiling path that previously hit `AssertionError: Only SM 10.x and
  11.x are supported`.
- Ran the 4-GPU vLLM no-MTP baseline:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda310/bin/python docs/qwen35_vllm_no_mtp_baseline.py \
  --tp 4 --gpu-mem 0.80 --max-model-len 8192 --max-tokens 32
```

- vLLM loaded successfully with `tensor_parallel_size=4`, logged the Qwen3.5
  Blackwell guard (`Auto-disabled DeepGemm ... E8M0 scale format causes
  accuracy degradation`), selected `CutlassFp8BlockScaledMMKernel`, skipped MM
  profiling, loaded weights in about 63s, and saved:

```text
docs/qwen35_vllm_no_mtp_baseline.json
```

- Baseline outputs are normal Qwen thinking text, not the RTP no-MTP garbage
  pattern (`ER`/`sp`). With `--max-tokens 32` the visible prefixes were:
  - `capital_france`: `Thinking Process:\n\n1.  **Identify the core question:** ...`
  - `translate_hello`: `Thinking Process:\n\n1.  **Analyze the Request:** ...`
- Next step: build and run unchanged-stress RTP no-MTP smoke with the local
  FP32-scale guard for Qwen3.5 on Blackwell.
