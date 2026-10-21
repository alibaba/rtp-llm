# 独立 Mock role

镜像仍使用固定入口 `sh /home/admin/whale_start.sh`。设置
`MOCK_BUNDLE_ROLE=mock` 后只启动一个 Mock JVM，由它承载多个逻辑 P/D
Engine；不启动 Master。未设置时沿用原有 Master＋Mock bundle 模式。
两种模式均要求显式设置测试开关 `RTP_LLM_MOCK_BUNDLE=1`。

Whale 只配置参数，不再配置 `MOCK_STARTUP_PY` 或 `MOCK_STARTUP_B64`。

| 环境变量 | 默认值 / 含义 |
|---|---|
| `MOCK_PREFILL_COUNT` / `MOCK_DECODE_COUNT` | 各 1；可为 0，总数必须大于 0 |
| `MOCK_GRPC_BASE_PORT` | 29130；按 P 后 D 连续分配端口；HTTP 控制端口为前一个端口 |
| `POD_IP` | 缺省解析 Pod hostname；必须是可通告的 IPv4 地址 |
| `MOCK_BLOCK_SIZE` | 1024 token |
| `MOCK_PREFILL_KV_POOL_BLOCKS` / `MOCK_DECODE_KV_POOL_BLOCKS` | 每个 Engine 各 1024 块 |
| `MOCK_DECODE_MAX_CONCURRENCY` | 每个 D 为 128 |
| `MOCK_EVENT_LOOP_THREADS` / `MOCK_COMPLETION_THREADS` | 各 4 |
| `MOCK_JAVA_OPTS` | `-Xms512m -Xmx2g`，支持带引号的 JVM 参数，不执行 shell |
| `FETCH_OUTPUT_STREAM` | 1；等待客户端取结果；0 开启 Mock 自动接续 |
| `MOCK_KMONITOR_ENABLED` | true；环境中的监控 role 别名由 Mock reporter 继承 |
| `MOCK_CACHE_STATUS_COMPACT_ENCODING` | false；显式开启后用紧凑快照编码 CacheStatus，减少 Java 临时对象；完整 keys、版本和 KV 容量语义不变 |
| `MOCK_PERFORMANCE_CONFIG_JSON` | 缺省使用镜像内 performance.json；显式值必须是完整 JSON 对象，block_size 与环境变量一致 |
| `FLEXLB_CONFIG` | 缺省 `{}`，使用 Mock 内置公式；也可提供匹配 Master 的配置 JSON 对象 |
| `JAVA_HOME` | `/opt/taobao/java` |
| `MOCK_BUNDLE_JAR_DIR` | 镜像内 tools/mock_bundle/jars；从中读取 mock.jar |

入口在 `MOCK_BUNDLE_RUN_DIR` 写出 performance.json、master-config.json、
endpoints.json 和 discovery.json；固定启动脚本自动为每次启动选择独立目录。
其中 discovery.json 只是本地记录，不会让 Master 自动使用文件发现。
VIP 注册和健康检查仍需部署侧配置；此入口不负责注册或注销 VIP。
Java 直接替换启动进程，接收容器终止信号。

独立 Mock Pod 的 2P3D、每引擎约 116M token 示例：

```text
MOCK_BUNDLE_ROLE=mock
RTP_LLM_MOCK_BUNDLE=1
MOCK_PREFILL_COUNT=2
MOCK_DECODE_COUNT=3
MOCK_PREFILL_KV_POOL_BLOCKS=113282
MOCK_DECODE_KV_POOL_BLOCKS=113282
FETCH_OUTPUT_STREAM=1
```

每引擎容量为 `1024 × 113282 = 116000768` token。模拟 KV 容量与 JVM
实际 heap 分别配置；Pod 资源由 Whale 的 mock role resource plan 配置。
