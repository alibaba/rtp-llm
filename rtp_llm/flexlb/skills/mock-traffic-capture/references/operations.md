# 触发、取数与恢复

在 Whale 核对 Frontend 与 ViT 的实际成员和镜像。用 ASI 核验一台 Frontend Pod 的 Kubernetes 名、集群、容器和 UID。后续查状态及下载固定用这台 Pod。工作盘先按计划时长和短批次实际体积预留空间。

## 启动及监控

```bash
capture=/opt/rtp-frontend-capture/capture.py
"$frontend_python" "$capture" run \
  --duration 40m --segment 2m --sort-mib 64 --work-dir "$capture_disk" \
  --frontend-pods "$frontend_pods" --vit-pods "$vit_pods"
"$frontend_python" "$capture" status "$batch"
```

`batch` 取启动返回的 ID。默认约 30 秒后开窗。可用 `--start-ms` 指定 15 至 240 秒内的 Unix 毫秒。段长支持 120 至 300 秒，最后一段可更短。时长可用 `s/m/h`。文本或全部使用 V4.1 内联几何时，`--vit-pods 0`。

`--keep-shards` 保留已验收原始分片，供离线 A/B 使用。它会增加全轮磁盘占用。默认在有序文件和验收报告提交后释放协调器的原始暂存。Frontend 和 ViT 的本地封存归档按该部署的留存计划独立清理。

观察 `armed_segments`、`verified_segments`、剩余磁盘及每段 `archive-checks.json`。状态达到 `ready` 或 `ready_with_gaps` 后，核对段数、窗口连续性和角色覆盖。汇报实际入库段、到达数、请求数、缺失率、分入口计数及图片引用数。缺失率以缺失请求数除以到达数。默认单段上限 1%，通过 `--max-missing-rate` 配置。

## 下载与校验

从 `status` 读取 `result` 和 `sha256`。用 ASI 文件传输下载该 Pod 上的 `result.tar`。传输只能走终端时，按偏移分块读取原始字节并 Base64 解码。每次核对实际字节数，重连后确认 Pod UID，再从已校验偏移续传。

```bash
offline=/opt/rtp-frontend-capture/offline.py
"$python" "$offline" import "$downloaded_tar" --sha256 "$result_sha256" \
  --shared-root "$shared_root" --name "$dataset_name.mckidx"
```

也可以在本机运行仓库内同名脚本，使用 Python 3.10+ 和 `zstandard==0.25.0`。导入校验归档 SHA-256、成品 SHA-256、文件头和请求数，输出数据集绝对路径、报告路径和字节数。Case 引用 `data/traffic/<name>.mckidx`，并传相同的 `--shared-root`。

## 部分批次恢复

批次失败时查看 `status.error`、`job.log` 和失败段的 `validation.json`。先确认协调任务已停止，再复制该批次的 `index/` 到隔离目录，依据清单恢复已验收的连续段：

```bash
"$python" "$offline" recover "$copied_index" "$new_result_dir" \
  --frontend-pods "$frontend_pods" --vit-pods "$vit_pods"
```

恢复目录输出 `replay.mckidx` 和标记为部分覆盖的 `report.json`。交付时注明实际窗口、缺段和缺失率。完整批次只保留已校验的成品、报表和各段验收记录。
