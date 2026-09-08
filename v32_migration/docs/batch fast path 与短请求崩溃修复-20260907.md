# batch fast path 与短请求崩溃修复

日期：2026-09-07

## 结论

本步只修改 Python hook/簿记层，没有修改引擎 C++、attention kernel 或 cuda graph。

- 原来会带崩全部 8 个 D rank 的 1,989-token C 纯短 batch 已修复；更短的 512-token batch 也通过。
- 512/1,989-token 交替混入一条 62,830-token offload 长请求时，8 条短请求和长请求全部完成，lossless admission/ring 正常，所有 D rank 存活。
- C 纯短 corrected TPOT 从 118.651ms 降到 **115.538ms**，减少 **3.113ms/token（2.62%）**。相对历史 A1 纯短 117.268ms 没有残留常驻税；低 1.730ms 属于跨服务运行漂移，不能解释为 C 比 A1 更快。
- C mixed 短请求 TPOT 从 117.863ms 降到 **116.366ms**，减少 **1.497ms/token（1.27%）**。两轮长请求和短请求 hash 与修复前一致。
- 最终 C mixed 相对最终 C pure 多 0.828ms/token。该差值包含真正的 mixed/offload 竞争，但本报告目标是证明 fast path 不破坏 mixed，不把它重新解释为精确混批税。

## 改动

文件：

```text
v32_migration/python/v32_capacity.py
v32_migration/python/v32_offload_hook.py
```

### 短请求入口早退

layer 0 每个 decode step 只根据现成的 host `sequence_lengths` 和 host block table 做一次 batch 判定。整批最大序列长度低于 `RTP_KV_OFFLOAD_MIN_SEQ=16384` 时，标记该 step 为 `native_only`。

纯短 step 的 61 层都直接调用原生 `_get_topk_paged`，不再执行：

```text
pre_topk
_bookkeep
process_layer_native_fast
process_layer
```

判定不读 GPU tensor，不引入 GPU→CPU 同步。后续层只读取一个缓存布尔值。

`_bookkeep` 保留 `2 + STG_BLOCKS` 宽度断言作为防御，但正常短请求不会再进入这里。修复不是把 34 列 buffer 动态缩短，而是把低于阈值的请求从 offload 簿记入口移除。

### native/managed 状态切换

从纯短 `native_only` 切回长请求 managed 模式时，旧的异步 `kvlens/khead` 不能继续按全局 step 外推。本步在 transition 时清空 stale step/plan/fast state，并从当前 host `sequence_lengths` 和 block table 播种首个 managed step。

这条切换缺少安全 host metadata 时立即抛错，不能继续使用旧状态。

### 异常边界

只有能够从 host 元数据证明整批仍低于 MIN_SEQ 的 resident batch，簿记异常才允许记录 ERROR 后降级到原生路径。

任何 offloaded 或不能证明 resident 的 batch 都 fail-closed，包括：

- admission generation 缺失或变化；
- host mirror 不完整；
- staging block 缺失；
- C++ lossless state 丢失；
- independent indexer pool 缺失；
- decode top-k/block-table shape 不匹配；
- lossless 模式使用 `V32_HOOK_LEVEL!=2` 或 `V32_SKIP_PROCESS=1`。

这些情况绝不静默走原生 attention。

### 其他热路径清理

- capacity 模式不再安装未使用的外层 shadow wrapper；
- managed `pre_topk` 只在 layer 0 调用；
- profiler env 开关只读取一次；
- steady admission 检查使用 generation-only ABI，不反复构造 host tensor wrapper；
- 每个新 fast step 先清旧 request key；
- stale admission mirror 使用 generation 条件释放；
- 删除已废弃 scorer import、常量和未使用 CUDA event。

## 稳定性验证

部署保持原实验配置：P TP2×DP4、D TP1×DP8、D 12GiB KV、`reuse_cache=0`、`enable_cuda_graph=0`，请求通过 `role_addrs` 固定到 D rank0。

| 场景 | 结果 | 输出 hash | 服务状态 |
|---|---:|---|---|
| 8 × 512 输入 / 1,024 输出，纯短 | 8/8 | `6b072883f83ebcdd` | P + 8 D 全活 |
| 8 × 1,989 输入 / 1,024 输出，纯短 | 8/8 | `e8c005cb1c47eee6` | P + 8 D 全活 |
| 1 长 + 4×512 + 4×1,989 | 9/9 | 短请求同上；长 `27887e7c2b6b7b51` | P + 8 D 全活 |

混合长度回归中，长请求为 62,830 输入 / 8,192 输出，明确触发：

```text
admission capped
4.78GB host mirror
ring load 934ms
```

worker status 连续 42 个两秒采样点显示 rank0 同时驻留 9 条请求。正式时间窗内没有 tensor width、CUDA、RDMA、disconnect、mirror missing 或 fail-closed 异常。

修复前 1,989-token pilot 的失败证据保留在：

```text
v32_migration/data/mixedbatch_tax_20260907/PILOT1989_C_pure_short_r1_*
```

原错误：

```text
RuntimeError: The size of tensor a (34) must match the size of tensor b (32)
```

## TPOT 对比

短请求 workload 与修复前完全一致：2,220 输入、1,024 输出、8 条、两波提交、固定 rank0。TPOT：

```text
(decode_service_us - decode_first_token_us) / (output_len - 1)
```

| 条件 | Round 1 mean | Round 2 mean | 合并 mean / p50 / p95 | 轮差 |
|---|---:|---:|---:|---:|
| 修复前 C 纯短 | 118.955 | 118.347 | 118.651 / 118.463 / 119.570 | 0.608 |
| 修复后 C 纯短 | 115.525 | 115.552 | **115.538 / 115.499 / 115.777** | 0.028 |
| 历史 A1 纯短 | 117.082 | 117.454 | 117.268 / 117.215 / 117.755 | 0.372 |
| 修复前 C mixed | 118.484 | 117.243 | 117.863 / 117.866 / 118.648 | 1.241 |
| 修复后 C mixed | 116.145 | 116.588 | **116.366 / 116.343 / 116.776** | 0.443 |

C 纯短前后差：

```text
115.538 - 118.651 = -3.113ms/token（-2.62%）
```

原先测得的 C 对 A1 `+1.383ms/token` 常驻税已经消失。绝对值低于历史 A1 是运行实例差异，结论只取“没有残留正税”，不宣称负税。

C mixed 前后差：

```text
116.366 - 117.863 = -1.497ms/token（-1.27%）
```

两轮均为 8/8 短请求和 1/1 长请求完成；短 hash 为 `47229f5fe1b12614`，长 hash 为 `27887e7c2b6b7b51`。长请求 corrected TPOT 为 113.191/113.681ms，未出现性能或正确性退化。

## 实施过程发现的问题

首次加入 fast path 后，纯短转回新长请求时复用了 stale 的异步 metadata，导致 KV 长度被按纯短期间的全局 step 外推，fail-closed 报 `lossy state missing` 并终止 D。该失败轮保留为：

```text
v32_migration/data/fastpath_20260907/FAILED_TRANSITION_C_mixed_r1_*
```

修复方式是上述 native→managed metadata 重播种。最终代码重新冷启服务并重跑稳定性与性能矩阵，最终正式轮均通过。

## 限定

本实验全程 `enable_cuda_graph=0`。结果只说明 eager Python/hook 路径的常驻税被消除，不能外推 cuda graph capture、graph break 或双图切换成本。cuda graph 仍应作为下一步独立验证。

## 证据

```text
v32_migration/data/fastpath_20260907/
```

正式性能轮：

```text
C_pure_short_r1/r2
C_mixed_r1/r2
```

稳定性回归：

```text
stability_pure_512
stability_pure_1989
stability_mixed_final
```

每轮包含 jsonl、client log、P/D GPU csv、D engine/Python log、host-memory、worker status、health、start/end epoch 和 exit code。最终 Python 文件及 SHA256 也保存在数据目录。未提交代码。
