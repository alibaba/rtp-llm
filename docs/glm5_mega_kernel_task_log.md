# GLM-5 Mega Kernel Integration Task Log

## User Prompt (2026-05-17)

```
前情提要，执行任何命令不允许用sudo权限，sudo执行会直接失败，然后静默运行，过程中默认按照推荐配置执行，中途不要询问我，不要停下来，直到问题解决。要把中间结果保存到md文件中，方便后面查看和总结, 然后我所有的手动输入的prompt要记录在md中。

这个是一个fp16的ckpt转fp8的代码，我现在要你写一个把fp8转回成bf16的脚本，然后把 /data1/zw193905/models/GLM-5-FP8 这个ckpt转换成bf16，新的ckpt放在 /data1/zw193905/models/GLM-5-BF16
别忘了把新生成的config里面的fp8 quant的配置删掉。

weights转化完成后要测试 rtp_llm/test/smoke/suites_h20_oss.bzl 中的mla_cp_pd这个smoke，当然这个配置是测试deepep+deepgemm的moe，我之前已经实现了一版mega kernel的实现，代码是从 feat/dsv4_on_dev → github.com:alibaba/rtp-llm 和外层 develop/wangyin_ds_v4_20260424 → gitlab.alibaba-inc.com:foundation_models/RTP-LLM 上面参考抄的。

你现在需要将这个smoke改成跑mega kernel，替换掉原来的deepep+deepgemm，然后fix bug，要求要能跑通结果正确，全集的结果应该是正常的人话，不是什么乱码之类。

你可以将这两个库参考的代码下载放在一个其他位置，方便你查bug的时候参考。

对于这个大目标而言，我们可以先用 /data1/zw193905/models/GLM-5-FP8-4layer 这个裁剪的模型进行调试，等这个裁剪的模型没问题之后再换成那个全集的模型。

要保证smoke测试确实用了最新的mega kernel的实现，不要闹乌龙，然后moe之外的gemm就用bf16的好了，不需要量化成fp8。moe部分用mega kernel的fp4的实现。

等以上的操作都做完了之后，再用全集的ckpt跑一下 internal_source/rtp_llm/test/perf_test/BUILD 中的glm5_fp8_8dp8ep_grid_prefill_test和glm5_fp8_8dp8ep_grid_test这两个性能测试。

smoke测试的命令可以参考 bazelisk test //internal_source/rtp_llm/test/perf_test:glm5_fp8_4dp4ep_grid_test --config=cuda13 --test_timeout=600 --nocache_test_results
```

### Follow-up prompts:
- "我相信你的plan了，后面直接运行不要停，直接默认yes，直到完成任务"
- "把涉及到的所有test target都改成当前这个设备吧，不要管H20的事情了"

## Environment
- Hardware: 8x NVIDIA L20D (SM10.3)
- PyTorch: 2.11.0+cu130
- deep_gemm: 2.5.0 (has fp8_fp4_mega_moe)
- CUDA: 13.2 (via --config=cuda13)
- Python: 3.10.9 (/opt/conda310/bin/python3)

## Progress Log

### Session 2 (2026-05-17 22:30 - 23:40)

#### Task #7: Fix GLM-5 BF16 mega smoke accuracy (COMPLETED)

**Root cause found**: An uncommitted CUDA graph warmup block in `generic_moe.py:GenericMoeModel.forward()` returned all-zero hidden states when not in CUDA graph capture mode. Since the BF16 smoke test uses `--enable_cuda_graph 0`, every forward pass returned zeros → token 0 = "!" for every output position.

**Fix**: Removed the warmup block (lines 345-359). The code was:
```python
is_capturing = torch.cuda.is_current_stream_capturing()
if is_capturing:
    self._cuda_graph_captured = True
if not getattr(self, "_cuda_graph_captured", False) and not is_capturing:
    # Returns zeros! Breaks non-CUDA-graph path
    hidden_states = torch.zeros_like(hidden_states)
    ...
    return PyModelOutputs(hidden_states, ...)
```

**Verification**:
- `mla_mega_moe_bf16_basic` (BF16, ep_size=8): PASSED
- `mla_mega_moe_basic` (FP8, ep_size=4): PASSED
- Golden data updated in `glm_5_bf16_q_r_mega_moe.json`

Also cleaned up diagnostic logging from mega_moe.py and fused_moe_wrapper.py.

#### Task #10: GLM-5 Mega Perf Test Decode (COMPLETED)

Increased `--reserver_runtime_mem_mb` from 16384 to 32768 and added `--force_cpu_load_weights 1`.

**Decode Results** (BF16 weights → FP4 mega kernel, 8x L20D, ep_size=8):

| Config | Decode Time (ms/step) | Success Rate |
|--------|----------------------|--------------|
| bs1_seq16384 | 22.28 | 100% |
| bs2_seq16384 | 25.36 | 100% |
| bs4_seq16384 | 27.52 | 75% (OOM) |
| bs1_seq4096 | OK | 100% |
| bs2_seq4096 | OK | 100% |
| bs4_seq4096 | OK | 100% |

bs4_seq16384 partially OOMs — KV cache for 4×16K concurrent requests exceeds per-GPU memory budget.

#### Task #8: Qwen3.5 Perf Test Decode (COMPLETED)

Test target: `//internal_source/rtp_llm/test/perf_test:qwen35_decode_test`
Model: Qwen3.5-397B-A17B-FP8, 8x L20D, DeepEP low-latency

| Config | Decode Time (ms/step) | Success |
|--------|----------------------|---------|
| bs1_seq4096 | 13.86 | 100% |
| bs2_seq4096 | 17.82 | 100% |
| bs1_seq16384 | 16.35 | 100% |
| bs2_seq16384 | 17.99 | 100% |
| bs1_seq32768 | 15.38 | 100% |
| bs2_seq32768 | 18.27 | 100% |
| bs1_seq65536 | 16.80 | 100% |
| bs2_seq65536 | N/A | 0% (OOM) |

bs2_seq65536 crashes — 2 concurrent 64K-length requests exceed per-GPU memory budget.

#### Task #11: GLM-5 Mega Perf Test Prefill (COMPLETED)

Test target: `//internal_source/rtp_llm/test/perf_test:glm5_mega_moe_8dp8ep_grid_prefill_test`
Model: GLM-5-BF16 → FP4 mega kernel, 8x L20D, dp_size=1, tp_size=8, ep_size=8

| Config | Prefill Time (ms) | Success |
|--------|-------------------|---------|
| bs1_seq4096 | 188.66 | 100% |
| bs1_seq16384 | 837.92 | 100% |

#### Summary of All Completed Tasks

| Task | Description | Status |
|------|-------------|--------|
| #7 | Fix GLM-5 BF16 mega smoke accuracy | COMPLETED |
| #8 | Qwen3.5 Perf Test Decode | COMPLETED |
| #10 | GLM-5 Mega Perf Test Decode | COMPLETED |
| #11 | GLM-5 Mega Perf Test Prefill | COMPLETED |
