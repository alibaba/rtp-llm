# 动态 Decode 注意力后端

使用说明。该功能**默认关闭**，不开启时下面这些逻辑都不参与，引擎仍走原来的静态注意力路径。

## 开启方式

```bash
--enable_dynamic_decode_backend true      # 或 ENABLE_DYNAMIC_DECODE_BACKEND=true
```

启动时先做一次精度检查，保留所有 rank 都通过的后端；随后对每个 CUDA Graph capture bucket 测速，按稳定选型策略选择后端建图。没有可用动态结果时，该 bucket 使用原有静态路径。

支持条件为 CUDA 上的 Python BF16 MHA 模型、开启 CUDA Graph、BASE/FP8 KV cache、单个 FULL KV cache group，且不使用 PP/CP、投机、多模态、MLA、headwise 注意力或 DeepEP low-latency。

当前已实现的配置回退包括：非 CUDA、没有 Python 模型、非 BF16、关闭 CUDA Graph、PP/CP、投机、多模态、MLA、headwise 注意力、DeepEP low-latency，以及非单个 FULL KV cache group。这些情况打印
`dynamic decode disabled, falling back to static path: ...`，关闭动态调度后进入原有静态路径。其他不支持的配置不保证回退；例如非 BASE/FP8 KV 可能在精度检查中报错。

如果检查执行了但没有任何后端通过，启动失败并报
`dynamic decode precision gate failed`。通过时的日志：

```
decode gate thresholds source=builtin id=h20-qwen3-20261030-v1 sha256=... ...
decode gate local rank=0/1 records_ok=True passed=['XQADecodeImpl', ...]
decode gate WORLD passed=['XQADecodeImpl', ...]
decode gate calibration_coverage={'tested_batches': (1, 3, 8), ...}
```

每个场景的判定明细在准入前打印，可以按 rank 区分是哪个场景、哪个后端没过：

```
decode gate rank=3/8 scenario=b1__s0_p0c64d4 backend=XQADecodeImpl layers=2 failures=[...]
```

## 阈值

内置了一套默认阈值。要换成自己的数值，把一整份 JSON 内联写进 `DYN_DECODE_GATE_THRESHOLDS`；它作用于当前进程实际使用的 profile（BASE KV 对应 BF16 那套，FP8 KV 对应 FP8 那套）：

下面是 **FP8 KV 阈值注入示例**，不是任意模型、硬件的通用推荐值；在 BASE KV 进程中注入同一 JSON，也会覆盖其 BF16 阈值。

```bash
export DYN_DECODE_GATE_THRESHOLDS='{
  "calibration_id": "h20-qwen3-my-run-v1",
  "target": {"sm_major": 9, "model_family": "qwen3"},
  "signal_split": 0.5,
  "budget_coefficient": 0.09,
  "roundoff_coefficient": 0.001953125,
  "gain_limit": 0.11,
  "cosine_limit": 0.95,
  "nrmse_limit": 0.33,
  "ulp_limit": null
}'
```

| 键 | 取值范围 | 含义 |
| --- | --- | --- |
| `signal_split` | (0, 1) | `R < signal_split × S` 时使用低信号预算，否则使用高信号判定 |
| `budget_coefficient` | >= 0 | 低信号 RMS 误差预算中参考贡献尺度 `S` 的系数 |
| `roundoff_coefficient` | >= 0 | 低信号 RMS 误差预算中 golden 输出 RMS `R` 的系数，并非绝对容差下限 |
| `gain_limit` | >= 0 | 相对 golden 方向的增益偏差绝对值上限，同时限制放大和衰减 |
| `cosine_limit` | (0, 1] | 高信号场景的最小余弦相似度 |
| `nrmse_limit` | >= 0 | 高信号场景的最大 NRMSE |
| `ulp_limit` | `null` 或 > 0 | 可选的高信号近似 BF16 间距归一化误差预算，并非严格可表示值距离 |
| `calibration_id` | 字符串 | 仅用于日志和报告 |
| `target` | `{sm_major, model_family}` | 仅用于日志和报告；与运行时不一致只打 WARNING，继续运行 |

`R` 为 golden 输出 RMS，`S` 为参考贡献尺度。低信号允许的 RMS 误差为 `budget_coefficient × S + roundoff_coefficient × R`；增益偏差和非有限数检查同时适用于高、低信号分支。

也可使用 `--dyn_decode_gate_thresholds '<完整JSON>'`，CLI 优先于环境变量。原始字符串经标准配置和进程序列化传给模型，仅在精度门执行时解析；未配置使用内置值，显式空字符串属于非法 JSON。覆盖阈值的来源日志记为 `source=config`。

七个数值键必须写全：缺键、多出未知键、数值越界或 JSON 语法错误都会在执行精度门时启动失败，不会把传入值和内置值混成一套没有标定过的参数。只支持内联 JSON，不读取阈值文件。性能选型固定使用 5% 提升门槛和 5% 候选聚类范围。

启动日志记录生效的完整阈值、来源、标定 ID 和 SHA256 摘要。建议填写 `calibration_id` 和 `target`，便于追溯标定来源及其适用条件。
