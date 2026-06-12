# GLM5 Eagle Acceptance Investigation - 2026-05-29

## User Prompt

> 前情提要，执行任何命令不允许用sudo权限，sudo执行会直接失败，然后静默运行，过程中默认按照推荐配置执行，中途不要询问我，不要停下来，直到问题解决。要把中间结果保存到md文件中，方便后面查看和总结,然后我所有的手动输入的prompt要记录在md中, 每一步执行完关键步骤状态都写到md文档里,有啥不清楚的读文档。分析一下sglang@/home/zw193905/sglang, vllm@/home/zw193905/vllm 的相关代码，重点查看下glm5 eagle相关的代码，然后再查看下这个库里的glm5 eagle相关的代码。目前我遇到的问题是开了eagle功能之后结果是正常的，但是接受率不太正常，目前GEN_NUM_PER_CIRCLE在3的情况下平均接受是2个，理论上应该是在2.7左右。所以我怀疑是eagle的实现代码有问题，是不是hidden states传的不太对，或者是glm5有一些特殊的draft采样的代码和我们的实现不一样。接受率应该和vllm的测试差不多对齐，vllm启动服务可以参考@/home/zw193905/RTP-LLM/github-opensource/docs/hidden_align 下的md文件。用@/home/zw193905/RTP-LLM/internal_source/rtp_llm/test/smoke/BUILD 里面的mla_mtp_mega_moe_cudagraph_pd_full_ckpt这个测试的query来跑一下vllm，看下实际的接受率是多少，然后看看当前这个smoke的query的接受率是多少，如果我们的接受率低，那就是哪里有bug，要排查问题。smoke测试命令可以参考bazelisk test //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt --config=cuda13 --test_timeout=1000 。在跑smoke测试之前先看下所有gpu是不是都闲置的，如果是的话再运行smoke，不然就稍微等一下。

## Continuation Prompt

> Another language model started to solve this problem and produced a handoff summary. Continue from the existing state without duplicating work. Key next steps from the handoff: validate `docs/glm5_eagle_vllm_acceptance.py`, check GPU idleness, run the vLLM acceptance baseline, run the RTP smoke, compare acceptance, and continue debugging hidden/position/cache/module semantics if RTP is lower.

## Continuation Prompt 2

> Another language model started to solve this problem and produced a summary of its thinking process. Use the summary to continue from the current state. Important facts from the handoff: vLLM baseline for the exact smoke query succeeded with mean acceptance length `2.4872417983`; RTP smoke timed out after startup and partial query execution; parse RTP partial logs, consider rerunning with larger timeout or narrower harness, and compare RTP against the measured vLLM baseline instead of the expected `2.7`.

## Follow-up Prompt

> 你是怎么确认目前的实现接受率在2.4到2.52

## Constraints

- Do not use `sudo`.
- Do not stop for questions; use recommended/default configuration where local context is sufficient.
- Persist key intermediate results and status updates in this file.

## Status Log

- 2026-05-29 16:02 CST: Started investigation from `/home/admin/zw193905/RTP-LLM/github-opensource` (resolved by shell as `/home/zw193905/RTP-LLM/github-opensource`).
- 2026-05-29 16:02 CST: Confirmed local paths exist: `/home/zw193905/sglang`, `/home/zw193905/vllm`, and current RTP-LLM checkout.
- 2026-05-29 16:02 CST: Existing worktree has unrelated modified/untracked files; will not revert or overwrite them.
- 2026-05-29 16:03 CST: Located target `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` in `internal_source/rtp_llm/test/smoke/BUILD`. It uses GLM5 full FP8 checkpoint, `sp_type=eagle`, `gen_num_per_cycle=3`, PD prefill/decode, prefill TP/EP/world=4, decode DP/EP/world=4, and debug envs `RTP_LLM_DEBUG_MTP_ACCEPT=1` plus `RTP_LLM_DEBUG_MTP_DECODE_DATA=1`.
- 2026-05-29 16:03 CST: Found prior related investigation logs (`glm5_eagle_debug_log.md`, `glm5_full_ckpt_precision_cudagraph_async_worklog.md`) and hidden-align docs/scripts under `docs/hidden_align/`; will use them as context without overwriting.
- 2026-05-29 16:06 CST: Live code check found no current uncommitted diff in the main GLM5/MTP files inspected. Current code already maps `glm_5_mtp` to `SpeculativeType.EAGLE`, keeps GLM5 MTP indexer/DSA enabled, masks MTP embeddings at `position == 0`, and makes the GLM target first-pass hidden handoff use final-norm hidden states.
- 2026-05-29 16:09 CST: vLLM GLM5 full checkpoint path exists at `/home/admin/zw193905/models/GLM-5-FP8`; config has `architectures=["GlmMoeDsaForCausalLM"]`, `model_type="glm_moe_dsa"`, `num_hidden_layers=78`, `num_nextn_predict_layers=1`, `hidden_size=6144`, `index_topk=2048`.
- 2026-05-29 16:09 CST: Smoke task info has 2 chat queries, both `max_tokens=1024`, `temperature=0.1`, `top_p=0.9`, `top_k=1`.
- 2026-05-29 16:09 CST: GPU pre-vLLM check: all GPUs are at 0% utilization; GPU 6 has about 1406 MiB residual memory and the rest are near idle. No compute-app rows were reported.
- 2026-05-29 16:09 CST: Resumed from handoff summary. Confirmed the investigation log and vLLM baseline script exist; current worktree has unrelated modified/untracked files, so only this investigation log/script will be touched unless a targeted fix is required.
- 2026-05-29 16:10 CST: `docs/glm5_eagle_vllm_acceptance.py` passes `py_compile`.
- 2026-05-29 16:10 CST: GPU recheck is not fully idle: `nvidia-smi` shows GPUs 0-3 at about 30.8 GiB each and 11-31% utilization, while GPUs 4-7 are at 5 MiB and 0% utilization. `nvidia-smi` process table and `pmon` report no running GPU processes, so this may be transient/stale driver accounting, but 8-GPU runs should wait until GPUs 0-3 settle.
- 2026-05-29 16:11 CST: GPU polling shows GPUs 0-3 increasing to about 150 GiB used with nonzero utilization; GPUs 4-7 remain idle. Proceeding with vLLM baseline on GPUs 4-7 using tensor parallel size 4, because acceptance should be independent of TP size for the same model/query/speculative settings. RTP smoke will still wait for the needed GPUs to be idle.
- 2026-05-29 16:12 CST: First vLLM attempt on GPUs 4-7 failed before inference: local vLLM extension loaded `/usr/lib64/libstdc++.so.6`, which lacks `CXXABI_1.3.15`. Checked `/home/zw193905/gcc14_env/lib/libstdc++.so.6`; it provides `CXXABI_1.3.15`, and `LD_LIBRARY_PATH=/home/zw193905/gcc14_env/lib:/opt/conda310/lib:$LD_LIBRARY_PATH` makes `import vllm._C` succeed.
- 2026-05-29 16:13 CST: GPU polling settled: all 8 GPUs are around 4 MiB used and 0% utilization. Retrying vLLM baseline on all 8 GPUs with TP=8 and the corrected `LD_LIBRARY_PATH`.
- 2026-05-29 16:22 CST: Second vLLM attempt completed engine/model/draft/KV/warmup initialization, then failed before query generation at `llm.get_metrics()` with `AssertionError: Stat logging disabled`. Patched the baseline script to pass `disable_log_stats=False` to `LLM(...)`.
- 2026-05-29 16:29 CST: Third vLLM attempt got past the metrics flag but failed during KV cache initialization with `Available KV cache memory: -40.28 GiB`. After shutdown all GPUs returned to about 5 MiB and no compute processes. Smoke prompt lengths are 26619 and 14013 tokens, so retrying with `max_model_len=28672` and `gpu_memory_utilization=0.95` instead of 32768/0.82.
- 2026-05-29 16:33 CST: Fourth vLLM attempt with 0.95 memory utilization failed earlier in worker device init: vLLM requires startup free memory >= requested utilization (about 254 GiB), but CUDA/NCCL context setup left some devices below that threshold. After failure, `nvidia-smi` temporarily shows large residual memory on all GPUs while reporting no compute processes; waiting for driver accounting to settle before retrying.
- 2026-05-29 16:43 CST: vLLM baseline succeeded with `LD_LIBRARY_PATH=/home/zw193905/gcc14_env/lib:/opt/conda310/lib:$LD_LIBRARY_PATH`, TP=8, `max_model_len=28672`, `gpu_memory_utilization=0.82`, `num_speculative_tokens=3`, `draft_sample_method=greedy`, and the smoke query file. Results saved to `docs/glm5_eagle_vllm_acceptance_20260529.json`.
- 2026-05-29 16:44 CST: vLLM measured acceptance: overall `num_drafts=823`, `draft_tokens=2469`, `accepted_tokens=1224`, acceptance rate `0.4957`, mean acceptance length `2.4872`, per-position acceptance `[0.8287, 0.4957, 0.1628]`. Query 0 mean acceptance length `2.4794`; query 1 mean acceptance length `2.4951`. This baseline is much closer to the observed RTP average around 2 than to the expected 2.7, but RTP smoke still needs to be measured directly.
- 2026-05-29 16:44 CST: GPU pre-RTP-smoke check is idle: GPUs 0-3 show 0 MiB/0%, GPUs 4-7 show 5 MiB/0%, and no compute-app rows are reported. Starting `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt --config=cuda13 --test_timeout=1000`.
- 2026-05-29 17:03 CST: RTP smoke timed out after about 1037s. Full-checkpoint service startup consumed most of the timeout: decode port became healthy around 16:54:08, prefill around 16:56:23, profile started, and the first chat completion request was sent at 16:56:24. The request did not complete before Bazel timeout. Need parse partial logs for any MTP accept data and rerun with a larger timeout or a narrower harness if full acceptance is required.
- 2026-05-29 17:03 CST: Resumed from the second handoff. Confirmed `SpeculativeSampler.cc` emits `[debug-mtp-accept]` lines with `accept_len=...`; next step is to parse the partial decode log from the timed-out RTP smoke before deciding whether another full smoke run is needed.
- 2026-05-29 17:06 CST: Parsed the timed-out RTP smoke partial decode log. The debug sampler budget captured 32 accept samples: `accept_len` list `[3,3,3,2,1,3,3,4,4,4,3,2,1,1,3,1,3,4,3,2,2,1,2,1,2,4,3,3,2,3,2,1]`, sum `79`, mean `2.46875`, histogram `{1:7, 2:8, 3:12, 4:5}`. The per-position draft/target match rates from those 32 samples are `[0.78125, 0.65625, 0.15625]`.
- 2026-05-29 17:08 CST: Confirmed the previous RTP smoke completed both serial queries before timing out in concurrent stress. Actual outputs were dumped for query 0 and query 1; the timeout occurred just after `[CONCURRENT_STRESS iter=1/8]` started.
- 2026-05-29 17:09 CST: Derived full-query RTP mean output tokens per iteration from dumped `aux_info`: query 0 `1024 / 413 = 2.4794188862`, query 1 `1024 / 401 = 2.5536159601`, overall `2048 / 814 = 2.5159705160`. Query 0 exactly matches the vLLM query 0 acceptance length (`2.4794188862`), and the overall value is close to vLLM overall (`2.4872417983`).
- 2026-05-29 17:07 CST: GPU check before rerun showed GPUs 4-7 still busy/residual (`18-225 GiB` on some devices and nonzero utilization), so held off running the 8-GPU smoke.
- 2026-05-29 17:08 CST: GPU polling reached idle: GPUs are `5-27 MiB`, `0%` utilization, no compute-app rows. Starting the same smoke with larger timeout to avoid the previous 1000s timeout: `bazelisk test //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt --config=cuda13 --test_timeout=2400`.
- 2026-05-29 17:18 CST: Rerun startup reached healthy ports: prefill port `17973` healthy at `17:18:28`, decode port `12467` healthy at `17:18:52`; profile started and the first query was sent immediately after.
- 2026-05-29 17:23 CST: Rerun completed: `//internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt PASSED in 905.6s` with `--test_timeout=2400`.
- 2026-05-29 17:23 CST: Rerun serial query outputs were dumped at `17:22:12` and `17:22:31`. The non-fatal concurrent stress section reported HTTP errors and `[CONCURRENT_STRESS] Result: FAIL`, but the test target still passed because the target sets `CONCURRENT_STRESS_FAIL_TEST=0`; this is separate from serial-query acceptance measurement.
- 2026-05-29 17:24 CST: Rerun RTP full-query mean output tokens per iteration: query 0 `1024 / 442 = 2.3167420814`, query 1 `1024 / 398 = 2.5728643216`, overall `2048 / 840 = 2.4380952381`. The first 32 debug accept samples in the passing run had mean `2.5`, histogram `{1:5, 2:12, 3:9, 4:6}`, and per-position match `[0.84375, 0.53125, 0.21875]`.
- 2026-05-29 17:24 CST: Post-run GPU check shows no compute-app rows; after a short driver-accounting settle, GPUs are at `727 MiB` on 0-3 and `5 MiB` on 4-7 with `0%` utilization.
- 2026-05-29 17:25 CST: Follow-up asks how the `2.4 ~ 2.52` acceptance range was confirmed. Answer basis: vLLM metric counters (`1 + accepted_tokens / num_drafts`), RTP serial output `completion_tokens / iter_count`, and RTP `[debug-mtp-accept] accept_len` samples.

## Static Alignment Notes

- vLLM maps `glm_moe_dsa` target configs to `DeepSeekMTPModel` for MTP speculative decoding. Its MTP layer uses `enorm(input_embeds)`, `hnorm(previous_hidden_states)`, `eh_proj(cat(...))`, masks input embeddings where `positions == 0`, runs the normal DeepSeek/GLM DSA decoder layer, returns pre-shared-head-norm hidden, and computes logits through `shared_head.norm + shared_head.head`.
- SGLang GLM4/GLM5-style NextN code uses the same `enorm/hnorm/eh_proj` structure and shared target embedding/head, but its visible `glm4_moe_nextn.py` applies `shared_head.norm` before returning hidden/logits. SGLang's DSA-MTP regression fixture uses EAGLE with `speculative_num_steps=3`, `speculative_eagle_topk=1`, `speculative_num_draft_tokens=4`, and acceptance thresholds of `2.7`.
- RTP current code now follows the vLLM-style first-pass hidden handoff for GLM: the main `GenericMoeModel.get_mtp_target_hidden_states()` returns `None`, so C++ keeps `model_output.all_hidden_states` for draft prefill. The draft `GenericMoeMTPModel` still exposes its pre-shared-head hidden buffer for recurrent draft steps.

## Conclusion

- For the exact smoke query and vLLM configuration (`num_speculative_tokens=3`, greedy draft), vLLM's measured mean acceptance length is `2.4872417983`, not `2.7`.
- RTP's measured acceptance is aligned with that baseline:
  - Timed-out run serial query outputs implied overall `2.5159705160` output tokens per iteration; first 32 debug samples averaged `2.46875`.
  - Passing rerun implied overall `2.4380952381` output tokens per iteration; first 32 debug samples averaged `2.5`.
- The current evidence does not point to a GLM5 EAGLE hidden-state handoff or draft sampling bug in RTP. RTP is in the same band as vLLM for this smoke query, and the observed gap from `2.7` is explained by the actual vLLM baseline plus workload/config differences from SGLang's DSA-MTP fixture.
- RTP metric naming can be confusing: `rtp_llm_sp_avg_accept_token_num` reports `accept_len`, which includes the guaranteed target bonus token and ranges from `1` to `gen_num_per_cycle + 1`; `rtp_llm_sp_avg_fix_accept_rate` is the draft-token acceptance rate `(avg_accept_num - 1) / gen_num_per_cycle`.
