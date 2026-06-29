# attention_backend_plan —— decode attention 后端选择 plan

存放 **decode attention 后端调度 plan**(`{capture bs 桶 → 后端名}`)的 autotune 结果。
与同级 `../cutlass_groupgemm/`、`../../triton_kernels/autotune_cache/configs/` 同类:
per-GPU 的 kernel / 后端选择调优产物,**入仓版本管理**(决策配置,非数据)。

## 命名约定

```
<device_name>/plan_<modelsig>_tp<tp>[_<workload>].json
例: NVIDIA_H20/plan_h32kv8d128p64_tp4_mooncake.json
```

- **device_name**:`torch.cuda.get_device_name()` 规范化(空格→`_`)。后端相对延迟随 GPU
  型号变(H20 的 plan 不能给 H800),**必按 device 分目录**——与 triton autotune_cache 一致。
- **modelsig**:`h<num_heads>kv<kv_heads>d<head_dim>p<page_size>`(per-rank,TP 切分后的 head 数)。
- **tp**:tensor parallel 度。
- **workload**:可选,业务分布标签(mooncake / azure / …);动态调度按各桶 kvlen 分布选后端时区分。

## 内容格式

```json
{
  "assignments": { "1": "XQAImpl", "8": "PyFlashinferDecodeImpl", "16": "PyFlashinferDecodeImpl" },
  "note": "..."
}
```

key = capture bs 桶(int,json 里为 str);value = 后端类名(`XQAImpl` / `XQADecodeImpl` /
`PyFlashinferDecodeImpl`)。

## 谁产 / 谁读

- **产**:动态 attention 调度器 warmup 时 profiling → `cached_plan(model, tp, device, build_fn, workload=…)`
  覆盖写到此(命中即跳过重测)。
- **读**:生产引擎按 plan 决定每个 cudagraph 桶 capture 哪个后端;离线评估器经 `Plan.load` 喂入做对比。
- **工具**:`Plan` 契约 + `cache_path` / `cached_plan`（即将随评估器代码提交到本仓）。

## 覆盖写,不累积

同 `(device, model, tp, workload)` 永远同一文件,覆盖写;要历史靠 git，不靠时间戳堆文件。
