# 离线验收与容量测量

使用 `--keep-shards` 留下的连续段。每段目录须包含 `shards/frontend/<Pod>/worker-<序号>-*.jsonl.gz` 及按需提供的 `shards/vit/<Pod>/vit-*.jsonl.gz`。计划示例中的 `shards` 相对计划文件路径解析。

```json
{
  "frontend_workers": { "frontend-pod-address": 4 },
  "vit_pods": 1,
  "sort_mib": 64,
  "max_missing_rate": 0.01,
  "segments": [
    {
      "start_ms": 1794275460000,
      "end_ms": 1794275580000,
      "shards": "1794275460000/shards"
    }
  ]
}
```

```bash
offline=rtp_llm/flexlb/mock-framework/scripts/traffic-capture/offline.py
uv run --no-project --with zstandard==0.25.0 "$offline" compare "$plan" "$new_output_dir"
```

`comparison.json` 包含全量 SQLite 与分桶归并的逐段 `ingest_seconds`、`work_peak_allocated_bytes` 和 `routing_reuse_loss_pp`，同时给出封装耗时、成品体积、请求数、图片数和 SHA-256。工作盘峰值统计各自的输出目录，Linux 上包含进程仍打开的匿名临时文件。输入归档和外部下载暂存不计入这一列。

全量复用验收要求两组请求数、媒体引用数及两种粒度的重复 token 数一致。再用相同 Case 读取成品，核对输出量语义、多模态展开及对应请求的路径。理论复用率的分母是全部入库请求的展开 token 数。引擎真实命中率还由容量、路由和请求时序决定。

长批次按计划容量逐档扩充历史，给每档保留相同的末段探针。记录探针提交时间、归并时间、待处理段、磁盘最低余量和收尾时间。上线前在实际峰值请求速率下检查积压能否消退。段归并跟不上时，先调工作盘、缓冲或批次容量，并复跑同输入验收。归档精度保持全量前缀匹配。
