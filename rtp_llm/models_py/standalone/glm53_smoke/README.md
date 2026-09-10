# GLM-5.3-Flash 四层性能与数值 smoke

后续优化固定使用这份入口、输入和 golden。模型取发布 checkpoint 的 **第 4、5、6、7 层（从 0 开始）**，组成 3 KDA + 1 MLA，四层均包含 routed MoE 和 fused shared expert。直接调用生产 `Glm53Flash`、权重加载器和 decoder layer；只缩减模型配置并重映射权重层号。MTP 关闭。

| 项目 | Prefill | Decode |
|---|---|---|
| 输入 | B8 × 131072 token | 每 rank B48，总 B384；历史长度 131072 |
| 并行 | TP8，MLA 专用 CP8，MoE EP8 | DP8，MoE EP8；Indexer 也是 DP |
| 执行 | eager 四层 | CUDA Graph replay 四层 |
| 参考目标 | 598 ms | 1.89 ms |
| 默认性能门限 | 不慢于目标 10% | 不慢于目标 10% |
| 数值门限 | 逐位一致 | 逐位一致，另检查 graph 与 eager |

计时使用 CUDA Event，每次取 8 rank 最大值，再取 9 次的中位数。包含完整四层和最后一次 deferred mHC post；不包含 embedding、最终 norm/head、输入准备、cache 恢复、golden 读写、profiler 导出或请求交付。它是算子优化 smoke，不能替代完整 45 层、真实 PD 传输或任务质量测试，也不能把结果称为整模型 TTFT/TPOT。

## 环境和固定配置

需要可用的 RTP CUDA 13.2 runtime、8 张同机 GPU 和发布 checkpoint。当前验证主机 CUDA Runtime 为 **SM103、148 SM**；以 Runtime 为准，不能采用 NVML 的冲突架构信息。每卡预算 180 GiB。

DeepGEMM 使用 `feat/glm53_flash` 的 `2f1e25133d8f6bbffd2a918ae2ef2979678233fe`，基于 `origin/opt_fp8` 并包含 GEMM/RS。已验证 x86 wheel SHA-256：

```text
51d2b0135848c0549e0d682c7e73af65216963a2099072faf9a8b1e2bdb06b82
```

基线见 `settings.py`：FP8 MoE、严格 FP32 router/Conv 权重、`mega_moe_fp8_se`、完整 shared 权重、本卡 token 路由、AG+GEMM/GEMM+RS、稳定 RS、Conv 输出布局融合、MLA CP8，f_a/g_a 本地低秩关闭。环境变量可显式覆盖优化开关；输出报告会记录实际值。变更权重或输入会使 golden 校验失败。

Decode `topk_v3` 的选中集合可以相同而输出顺序不同，因此这份基线打开 `DSV4_INDEXER_TOPK_CANONICALIZE=1`，在 Prefill 和 GLM compressed Decode 两端排序，保留 `-1` padding。否则 MLA 的累加顺序会改变，无法建立严格可重复的 golden。

## 运行

先配置已构建的 RTP Python/native runtime 与 DeepGEMM wheel。以下 `python` 必须是该 runtime 的解释器；输出和 golden 放在有足够空间的 `/dataN` 卷。正常回归不传 `--write-golden`。

```bash
python -m rtp_llm.models_py.standalone.glm53_smoke.launch \
  --run-dir /dataN/user/glm53/prefill_verify -- \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m rtp_llm.models_py.standalone.glm53_smoke \
  --phase prefill --checkpoint /dataN/user/models/GLM-5.3-Flash \
  --output /dataN/user/glm53/prefill_verify \
  --golden /dataN/user/glm53/golden/prefill --trace

python -m rtp_llm.models_py.standalone.glm53_smoke.launch \
  --run-dir /dataN/user/glm53/decode_verify -- \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m rtp_llm.models_py.standalone.glm53_smoke \
  --phase decode --checkpoint /dataN/user/models/GLM-5.3-Flash \
  --output /dataN/user/glm53/decode_verify \
  --golden /dataN/user/glm53/golden/decode --trace
```

首次建立经审查的新基线时，指定新的空 golden 目录，并增加 `--write-golden`。已有 golden 不会覆盖；生成过程中失败的目录没有 `COMPLETE.json`，不能用于回归。golden 不进入 Git。

Launcher 在启动前对全部 8 卡逐秒采样，至少完整 5 秒均小于 1024 MiB。失败时换空闲主机；不要停止他人的任务。运行时记录显存和外部进程，超过预算或计时窗口有外部 GPU 进程会判失败。命令结束或被取消后，只关闭带本次唯一标记且进程启动时间仍匹配的进程，保存至少 5 秒清理证据。每次使用新的运行目录。

Bazel 入口为 `//rtp_llm/models_py/standalone:glm53_four_layer_smoke`；纯 CPU 合约测试为 `:glm53_four_layer_smoke_contract_test`。

## Golden 合约

- `inputs_rankN.pt`：完整 input IDs。Prefill 初始 cache 全零；Decode 的 BF16 KV、FP8 indexer pool、FP32 recurrent/Conv/compressor 历史逐字节保存。Decode 历史是有限值的固定合成状态，**不是 PD Prefill 生成的历史**。
- `rankN.pt`：四层全部本卡 token 的完整 BF16 mHC/hidden 输出；每层另保存通过同一 final norm 和全词表 LM head 做 FP32 投影的诊断 logits。Prefill 取每 rank 16 个固定位置（含边界和尾部），Decode 取全部 48 行。诊断读出不参与计时；完整 Prefill 全 token 全词表读出会超过 TB，因此不保存。
- `rankN.json`：文件 SHA-256，以及原 checkpoint 层号、实际加载的每个权重张量形状/dtype/SHA-256、输入和状态文件 SHA-256、拓扑与位置。
- `COMPLETE.json`：全部 8 rank 数值和性能通过后才生成。

Prefill golden 约 129 GiB，Decode 输入历史与 golden 合计约 70 GiB。校验采用有界分块，并保留完整 hidden，不仅比较抽样 logits。首次运行内部重复两次，新进程回归还会读取固化文件比较。图回放前恢复初始 hidden/residual 和 cache；mHC post 会原地写 residual，省略其恢复会造成伪精度差异。

默认 relative L2 与 max-abs 门限都是 0。relative L2 为 `||actual-reference||₂ / ||reference||₂`。优化若改变舍入路径，应先定位差异，再显式选择经评审的门限；不要重新生成 golden 来消除失败。

每次输出 `result_rankN.json`、`input_manifest_rankN.json`。加 `--trace` 只导出 rank 0 的一份 `prefill.json` 或 `decode.json`；性能样本在 profiler 外采集。采样先丢弃一次完整 profiler 热身，仅对 rank 0 开启 profiler，其余 rank 正常执行，并在 CPU 上同步启动，再验证 GPU 跨度位于独立计时中位数的 ±20% 内；最多尝试 3 次，失败则保留诊断文件并报错，不输出误导性的典型 timeline。查看 timeline 前需确认 3 个 KDA 层、1 个 MLA 层、4 次 MoE，且无首次编译或外部进程干扰。
