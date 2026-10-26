# Mock ViT 图片链路

Mock ViT 是 JavaMockEngineCluster 的 VIT 角色，和 P/D 共用入口、JVM、Netty 与 RPC 执行资源。每个 V worker 的服务地址、身份、缓存、队列、并发和 timing 状态独立。当前实现对齐 main 的通用 URL 图片协议；模型 forward 使用确定性 CPU 数据模拟，图片解码和 Qwen2/2.5-VL token 数量计算实际执行。

## 请求顺序

1. frontend 携带图片 cache affinity keys 向 Master 发送 `vit_route_only` 请求；这个阶段不创建 P/D admission。
2. frontend 对选定 ViT 调用 `GetMultimodalHashes`。先只探测 key；缺失时提交对应图片，得到每张图的行 hash、长度及 worker instance。
3. frontend 用 hash 展开图片标签，携带展开后的 seq_len、KV block keys 和 selected_vit 再请求 Master。原始 prompt tensor 保留，GenerateInputPB 使用展开后的 token_ids 和 multimodal_token_layout。
4. Master 核对 ViT 的 group/worker generation；失效时返回 8408，frontend 丢弃展开结果并最多重选一次。ViT-only 取消或响应丢失不能取消同请求号的 P/D admission。
5. P 调用同一个 ViT 获取 embedding，重新计算行 hash，并校验 layout 长度和 frontend token_ids。D 沿用 P/D 流程。BATCH dispatcher 必须把已选择的 ViT 地址与 P/D 一并下发。

## 启动与连接

```bash
java -jar <mock-engine.jar> \
  --mock.engine_config <engine-config.json> \
  --mock.calibration <calibration.json> \
  --mock.endpoint_file <runtime/endpoints.json> \
  --mock.n_prefill 1 --mock.n_decode 1 --mock.n_vit 2
```

`--mock.n_vit` 默认 0；启用时 V 默认和 P/D 共 JVM，端口连续排列为 P、D、V。需要 prepared V4.1 输入时设置 `--mock.vit_model_path <模型元数据目录>`。real VitConfig、每 worker 的缓存字节预算继续从 engine-config 读取；Mock 假特征宽度、输出保护和计算耗时从 calibration 的 `mock.vit`、`mock.forward.vit` 读取。字段语义统一见 [引擎配置](../../online_eval/docs/architecture/engine-configuration.md)。

各角色保留真实 gRPC 接口，P 仍通过地址访问已选中的 V；共 JVM 不绕过序列化、deadline、取消或 RPC 容量与抖动。缓存预算按 worker 独立计算；实际 heap、GC 和 JVM 故障由同一进程承担，不模拟物理 GPU 内存隔离。

Master 的 `MODEL_SERVICE_CONFIG.role_endpoints` 包含 `vit_endpoint`；动态发现文件统一由集群原子写入。发现地址使用 HTTP 端口约定，gRPC=HTTP+1；集群的统一 HTTP 控制口为 base_grpc_port-1，不为每个逻辑 worker 额外开 HTTP server。

`/snapshot`、`/effective_config`、`/health` 包含 P/D/V。V 的通用 running/waiting 计数分别表示 active/queued 图片计算任务，不把 GPU 队列中的同一任务重复相加；GPU 队列深度在 vit.gpu_queued 中独立展示。`/add_engine {"role":"vit"}`、`/remove_engine`、`/stop_engine`、`/start_engine` 使用与 P/D 相同的目标选择器。优雅移除先关闭新请求门禁并撤下发现地址，再等待已接受的工作和 RPC 回包完成，最后释放 V 缓存与执行资源；失败绑定回滚实例，发现写入失败回滚变更。逻辑 stop/start 保留已有缓存和 worker 身份；移除后重建或重新启动 JVM 创建新身份和空缓存。

`/set_perf` 对 V 接受 `mock.timing`、`mock.rpc`、`mock.forward.vit`，不能更改 P/D 性能或 real 容量配置。`/metrics` 对 V 按角色聚合；`?per_engine=true` 使用与 P/D 一致的 engine_name、role、grpc_port、engine_ip、engine_incarnation 标签。

## 缓存、状态与错误

embedding 和 hash 元数据分别按字节限制 LRU；embedding 淘汰后 hash 可继续路由，再次取特征会得到相同值。hash 缓存关闭时冷请求仍返回本次计算结果。相同 key 并发请求共享计算，一个等待方超时不取消其他等待方。

普通 `GetCacheStatus` 返回空缓存状态；`need_cache_keys=true` 才返回 worker instance、hash keys 和 CPU embedding keys。`GetWorkerStatus` 对齐 main：VIT、alive、status_version=1。缓存目录与 WorkerStatus 分开拉取。

坏图片、非法 key 和预处理参数返回 INVALID_ARGUMENT；计算队列已满返回 RESOURCE_EXHAUSTED；等待超时返回 DEADLINE_EXCEEDED。Mock 另有输入、输出、解码像素及队列上限，这些是测试资源保护，不代表真实模型容量。

## 验证与边界

`MockVitServiceTest` 覆盖真实 gRPC、冷查/补齐、特征与 hash 一致、淘汰、并发去重、超时隔离、错误输入和 P 校验。`VitCacheDirectoryTest` 覆盖 Master 的目录、实例变化及选路。frontend 的 token expansion 测试校验原 tensor 与 wire 输入的关系。

在真实 frontend 运行环境中执行：

```bash
python3 rtp_llm/flexlb/tools/mock_bundle/vit_frontend_smoke.py \
  --frontend http://127.0.0.1:8000 --vit-grpc 127.0.0.1:62000 \
  --require-reuse --output /path/to/run/vit-smoke.json
```

该脚本覆盖文本、单图、多图、重复图、冷热缓存及并发请求，检查 usage、input_len、图片 token 数和前缀命中。分别用 NON_BATCH 和 BATCH Master 运行。Mock 当前把命中 token 数写入 `prefix_len`；`reuse_len` 没有完整的阶段归因，不能单独用它判断是否命中，应结合 P/D 的 cache-key 计数。

支持 URL IMAGE 与 inline gRPC FP32，以及配置真实模型元数据后的 V4.1 prepared patches/BF16。尚未模拟视频、音频、通用预处理 tensor、模型特定位置编码、GPU 缓存 tier、RDMA 租约或真实内容审查。未配置 V4.1 模型时，packed 输入会显式拒绝。GreenNet 接口仅代表真实实现的 disabled/noop provider。固定延迟和线程池不代表真实动态 batching 性能。

`develop/dsv41_ced_release` 的 `metadata_only` 与 main 的 `support_rdma` 占用同一字段号，旧版返回逐图 `MultimodalOutputsPB`，不能在同一监听地址自动猜测协议。旧路径适配和线上事故复现应在独立分支、独立进程中验证；现代通用图片链路通过不等于 V4.1 事故已复现。

## Whale bundle

bundle 使用 `MOCK_VIT_COUNT` 启用 V，并把数量传给同一个 Mock JVM，不启动额外 V JVM、不拼接第二份服务发现文件。模型元数据路径为 `MOCK_VIT_MODEL_PATH`。`identity.json` 记录 V 数量和共用的 Mock PID，日志在 `mock.log`；旧 Master 协议不支持现代 ViT。

Whale 的 `per_engine` 部署按 `ROLE_TYPE=PREFILL|DECODE|VIT` 启动一个 worker。V launcher 准备模型元数据和 JVM 参数，最终同样调用 JavaMockEngineCluster，数量为 0P/0D/1V；P/D/V 的 JVM 和 Pod 边界由部署副本决定。统一入口也允许本地显式启动 V-only JVM，用于进程故障验证，无需第二套 V bootstrap。这里描述的是启动契约，本地回归不代表 Whale 平台部署已验收。

### 隔离 frontend 测试镜像

`whale_frontend_fixture.py` 是通用图片协议的 CPU 验收入口，要求显式设置 `RTP_LLM_MOCK_FRONTEND=1` 和 `MOCK_FRONTEND_SOURCE_ROOT`。它只使用所声明的打包源码及 protobuf，不会下载旧版 frontend。原生 `libth_transformer_config.so` 必须与打包源码兼容；使用原生启动 supervisor。

该入口在运行目录生成小型 Qwen2-VL 模型配置和确定性 tokenizer，默认 HTTP 端口 20550、KV block size 64，继承部署的 `MODEL_SERVICE_CONFIG` 连接真实 Master。它只生成模型元数据，不加载真实权重。配套 Master/P/D 必须同样使用 block size 64；此配置用于协议和缓存正确性验证，不能沿用原部署的容量或性能结论。

多 worker 验收应先自然预热，再检查热前缀；大量空闲 P 的轮转可能使少量重复请求仍落到冷 worker。应同时核对 `prefix_len`、P 的 `cache_key_hits`、D 的 `decode_reuse_blocks` 和完成后的请求/响应缓冲清理，不能只以 HTTP 200 代表缓存验收通过。

## main V4.1 prepared images

V 使用 `--mock.vit_model_path <模型元数据目录>`；bundle 使用 `MOCK_VIT_MODEL_PATH`。
读取真实 config.json 的 text/vision 元数据，校验与 main `V41ImageProcessorConfig.identity` 相同的处理器身份。
整批先检查 schema、BF16 patch shape、降采样网格、token types、跨度和资源上限，再提交 CPU 计算。
输出 `[image span rows, text hidden size]` BF16；包含 start/newline/end 行。
P 校验 canonical token metadata/image mask 和返回行数，保留 frontend 的 canonical IDs，不用 embedding hash 覆盖它们。
text-only V4.1 metadata 不触发 ViT RPC。

模型 forward 为确定性 CPU 模拟。非空 content identity 采用 main vision encode cache 的信任语义；空 identity 不命中或填充缓存。
KV key 使用 main C++ 的长度前缀字节身份及模型 namespace：默认 full `0x4453563431494631`，
`DSV41_SWA_BOUNDED_REPLAY=1` 使用 `0x4453563431494231`；可显式用 `MOCK_CACHE_KEY_HASH_SEED` 对齐其他模型 namespace。
完整物理 block 产生 key，不能在图片中间结束复用，也不能越过缺少身份的图片声明命中。
此实现没有覆盖真实 GPU 计算、RDMA 和多 pool 的数据布局/容量等价性。

## 生产组件边界

Mock 复用生产 frontend、Master 和 protobuf 协议，仅替代 ViT 模型计算及 P/D 引擎执行。
图片 BATCH 要求 dispatcher 保留已选 ViT 地址；V4.1 图片 cache key 要求 frontend 与 C++ 使用相同身份编码。
图片 ViT-only 查询必须携带非空、合法的图片 affinity keys，Master 保留 main 的严格校验。
main 的 prepared V4.1 frontend 路径目前漏传这些 keys；本分支不通过放宽 Master 校验规避调用方缺口。
BATCH 地址和 frontend/C++ 身份编码修复与 Mock 功能分开提交和验证。

协议集成 smoke 使用 `vit_protocol_smoke.py --master host:grpc-port --control http://host:mock-control-port --processor-identity <sha256> --output result.json`。
它使用生成的 protobuf，对真实 Master 调度、Mock ViT、P/D 执行及排空做检查；必须分别启动 NON_BATCH/BATCH 环境。
覆盖 URL 图片、V4.1 prepared 图片、text-only prepared metadata。此客户端不覆盖真实 frontend 的 HTTP 入口、原生扩展加载或真实图片预处理。
当前脚本保留 prepared V4.1 的空 keys 调用，以暴露上述调用方缺口；该 case 会被 Master 拒绝，不能宣称整套协议验收通过。

## Mock ViT KMonitor

`MOCK_KMONITOR_ENABLED` 控制 Whale 的监控开关，launcher 统一生成 `--mock.kmonitor true|false`。本地 Java 默认关闭监控。V 的队列、缓存、计算和 RPC 指标保持各自口径；status/cache polling 不计为图片业务 QPS。标签携带 V 的 engine、engine_ip、engine_port、generation，物理 Pod IP 与逻辑引擎地址分开；同 JVM 多 V 不得只按 host 聚合后判断某个 worker 的行为。

`source=mock_vit` 的指标与 frontend 同名指标分开查询。`py_rtp_vit_qps/success_qps/error_qps` 表示 embedding 请求入口和终态；`mock_vit_computations_total` 表示实际计算次数，缓存命中不增加它。forward/RPC timing 指标独立记录缩放后的计划时间、停顿与调度迟到，不能替代实际墙钟延迟。
