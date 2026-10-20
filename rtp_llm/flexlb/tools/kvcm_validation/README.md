# KVCM 远程对接实验

`run_smoke.py` 复用 online_eval 的 `EnvManager`、`EngineOps` 和进程清理底座，
运行 1P＋1D、NON_BATCH Master 与真实 KVCM。它是协议对接实验入口，尚未注册为默认场景或 CI suite。

先按 online_eval 的[编译与运行](../online_eval/docs/development/build-and-runtime.md)准备 Java 21、
Python 依赖及 Mock JAR。KVCM 包使用 `deps/http.bzl` 的
`remote_kv_cache_manager_server_cuda130` URL 和 SHA256，校验后解压至本租约目录。
它只需 CPU；包名中的 cuda130 不表示测试申请 GPU。

替换 Master 时，从指定源码 SHA 独立构建 `flexlb-api`，通过 `FLEXLB_FT_MASTER_JAR`
和 `FLEXLB_FT_MASTER_SOURCE_COMMIT` 传入。不要把不同分支的 reactor 模块混编。

从 `rtp_llm/flexlb` 运行：

```bash
export JAVA_HOME=/path/to/java21
export PATH="$JAVA_HOME/bin:$PATH"
export FLEXLB_FT_MASTER_JAR=/path/to/master.jar
export FLEXLB_FT_MASTER_SOURCE_COMMIT=<full-sha>
export FLEXLB_FT_MOCK_BASE_GRPC_PORT=<leased-mock-grpc-port>
export FLEXLB_FT_MASTER_HTTP_PORT=<leased-master-http-port>
export FLEXLB_FT_MASTER_MANAGEMENT_PORT=<leased-master-management-port>
python tools/kvcm_validation/run_smoke.py \
  --kvcm-dir /path/to/extracted-kvcm \
  --out /path/to/new-output \
  --port-base <leased-kvcm-base>
```

KVCM 占用 base 到 base+3 四个端口。它与 Mock、Master 端口均须落在当前租约内且互不重叠。
脚本拥有 KVCM 进程，启动前拒绝已占用端口；不要提前启动同一服务。

实验依次验证注册与快照、完整请求、KVCM 可查询 key、强制淘汰后位置消失、
KVCM 停止期间请求完成、重启后快照重建，以及正常退出后的 HOST_DOWN 清理。
HTTP 200 不代表通过；检查业务码、逐 key 查询和 `result.json`。
`protocol.json` 保存原始请求与响应，环境目录保留 Master 制品身份、配置和 JVM 日志。

该实验使用 KVCM 的 `ST_EVENT_REPORT_L1P5` 协议和本地内存索引。
NFS storage 仅提供独立测试配置，不执行 block 数据写入；测试不覆盖跨引擎 KV 搬运、
20 亿 key 容量、性能、生产资源池网络或 Whale 部署。

## 元数据容量评估

`run_capacity.py` 单独评估真实 KVCM，不启动 Master 或推理 JVM。
有界批次的 Java 负载生成器模拟多个上报节点，共用一个模型 instance，每个 key 只有一个持有位置。
它避免在发送端再保存一份全量 key map；逻辑上报节点数不能表述为真实 P/D Pod 数。

先用 Java 21 编译同目录的 `KvcmCapacityLoad.java`，再运行：

```bash
javac tools/kvcm_validation/KvcmCapacityLoad.java
python tools/kvcm_validation/run_capacity.py \
  --kvcm-dir /path/to/extracted-kvcm --out /path/to/new-output \
  --port-base <leased-base> --cpus <leased-cpus> \
  --memory-budget-gib <process-address-space-limit> \
  --rss-stop-gib <rss-stop-threshold> \
  --reporters <logical-reporters> --concurrency <http-concurrency> \
  --backend-capacity-mib <local-backend-quota-in-MiB> \
  --checkpoints <ascending-comma-separated-key-counts>
```

`max_key_count` 和 local backend 字节配额分别生效。`--backend-capacity-mib` 通过
`meta_storage_backend_config.storage_uri` 的 `capacity` 参数设置，单位 MiB；省略时使用服务端默认值。
检查初始化日志和回读的 instance group，不能把默认配额触发的 `EC_NOSPC` 当成宿主机内存极限。

每级保留服务端常驻 key 指标、业务响应计数、位置查询、RSS 和负载耗时。
停写后最多等待 `--metric-settle-seconds`（默认 60 秒），要求常驻计数精确等于目标；
指标尚未更新不能作为通过，也不能仅凭一次滞后采样断言丢 key。
`TARGET_REACHED` 表示所声明目标完成；`RESOURCE_BOUND` 表示预算边界，`ERROR` 保留失败。
20 亿内存投影是按当前字节/key 外推，不是 20 亿实测通过；快照恢复、共享 key 的多位置放大、
并发删除与推理链路性能需要独立验证。采样查询正确不等于对全部 key 做逐条读取验证。
