# Whale Mock 配置与环境变量

## 寄生 bundle

默认配置在 `tools/whale_mock/bundle.yaml`。一个 Pod 内有 Master JVM 和 Mock JVM，Mock JVM 管理多个逻辑 P/D 引擎。

| 变量 | 作用 |
|---|---|
| `RTP_LLM_MOCK_BUNDLE=1` | 必填的 test-only 开关 |
| `START_PORT` | Master HTTP 端口；management 与 Mock control 从配置偏移派生 |
| `POD_IP` | 对外通告的 Pod 地址；不能是 loopback 或 `0.0.0.0` |
| `FLEXLB_CONFIG` | 传给 Master 的完整配置，也是 Mock 公式投影的来源 |
| `MOCK_BUNDLE_OVERRIDES_YAML` | 覆盖逻辑 P/D 数、heap、block size、KV pool 和 D 并发 |
| `MOCK_PERFORMANCE_CONFIG_JSON` | 完整性能 JSON；覆盖镜像内默认性能文件 |
| `MOCK_EOS_CONFIG_JSON` | 可选 EOS 对象；设置后覆盖 performance 内 EOS |
| `FETCH_OUTPUT_STREAM` | `0` 自动接续、不等 frontend Fetch；`1` 等待完整 FetchResponse |
| `MOCK_EVENT_LOG_ENABLED` | 是否保存请求级 JSONL；生产式观测默认依赖 KMonitor |
| `MOCK_PREFILL_HIPPO_ROLE` / `MOCK_DECODE_HIPPO_ROLE` | 显式监控 role 别名，值按原样使用 |
| `MOCK_BUNDLE_LEGACY_MASTER=1` | 使用镜像内 pin 的 legacy Master |
| `MOCK_BUNDLE_FILE_DISCOVERY=1` | legacy 模式显式使用 bundle 文件发现适配 |

`MOCK_BUNDLE_OVERRIDES_YAML` 只接受实现列出的字段，所有数值必须为正整数。P/D 数量变化后必须同时检查 Pod CPU/内存、端口数、Mock heap、每引擎 KV 容量和 D 并发；只增逻辑实例不会自动增加 Pod 资源。

性能 JSON 的 `block_size` 必须与 bundle 的物理 block size 一致。P/D 使用不同有效 block size 时同时配置 `prefill_block_size`、`decode_block_size` 和对应 pool blocks，不能用路由层旧缓存块参数替代物理 KV 块。

## 独立 P/D Pod

| 变量 | 作用 |
|---|---|
| `FLEXLB_MOCK_WHALE=1` | 必填的独立 Pod 开关 |
| `ROLE_TYPE` | `PREFILL` 或 `DECODE` |
| `POD_IP` | 平台通告地址；缺省时入口尝试取 Pod 内首个地址 |
| `START_PORT` | HTTP 健康/控制端口；gRPC 为 `START_PORT+1` |
| `MOCK_PERFORMANCE_CONFIG` / `_JSON` | 性能配置路径或完整 JSON，二选一 |
| `MOCK_MASTER_CONFIG` / `_JSON` | 与 Master 匹配的配置路径或完整 JSON，二选一 |
| `FETCH_OUTPUT_STREAM` | `1` 等待 Fetch；`0` 自动接续 |
| `MOCK_KMONITOR_ENABLED` | Whale 默认为 true；缺少真实 adapter 时启动失败 |
| `MOCK_EVENT_LOG_ENABLED` | 可选请求级事件日志 |
| `MOCK_RUN_DIR` | 写入运行时配置和可选日志的目录 |
| `MOCK_BIND_HOST` | gRPC 监听地址，默认 `0.0.0.0` |

Whale 启动命令应为 `sh /opt/flexlb/start.sh`。配置 JSON 放在环境变量或挂载文件中，不要把带引号的多段 shell 填入平台 `cmd`。

## Whale 模板字段

- P 与 D 使用独立角色、资源模板、VIP 域和副本数。
- `resource_plan.meta_tag_list` 必须初始化为空列表。
- 最终镜像由顶层 `image_infos` 选择；资源槽位里的 package 信息可能被覆盖。
- 健康检查、容忍配置和资源池必须与目标池匹配，即使 Mock 不申请 GPU。
- 以最终 Carbon plan 为准核对镜像、环境、端口、健康检查和角色数量。

## 监控标签

Mock 指标使用引擎兼容的 `rtp_llm_*` 名称，并保留 `hippo_app`、`hippo_role`、`host_ip`、`container_ip`、`engine`、`engine_port`、`role`、`generation` 和 `backend=mock`。寄生 bundle 的多个逻辑实例必须按 `engine` 或 `engine_port` 拆分，不能把 Pod 总和当作单引擎值。

Prefill `context_tps` 按有效计算 token / 对应 batch execution 时间统计；`context_tps_with_cache` 用包含复用的输入 token 及其对应执行时间。它们不是墙钟吞吐；墙钟指标是 `context_wall_tps` / `context_wall_tps_with_cache`。Mock 数值用于验证配置与调度，不代表实际 GPU kernel 性能。

真实规模的部署校准必须执行 [生产配置与监控对齐](production-alignment.md)，不能只修改 P/D 数量或复制一段性能公式。
