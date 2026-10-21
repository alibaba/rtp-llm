# Mock ViT 图片链路

Mock ViT 是独立 Java gRPC 进程，和 P/D 共用 mock engine JAR。当前实现对齐 main 的通用 URL 图片协议；模型 forward 使用确定性 CPU 数据模拟，图片解码和 Qwen2/2.5-VL token 数量计算实际执行。

## 请求顺序

1. frontend 携带图片 cache affinity keys 向 Master 发送 `vit_route_only` 请求；这个阶段不创建 P/D admission。
2. frontend 对选定 ViT 调用 `GetMultimodalHashes`。先只探测 key；缺失时提交对应图片，得到每张图的行 hash、长度及 worker instance。
3. frontend 用 hash 展开图片标签，携带展开后的 seq_len、KV block keys 和 selected_vit 再请求 Master。原始 prompt tensor 保留，GenerateInputPB 使用展开后的 token_ids 和 multimodal_token_layout。
4. Master 核对 ViT 的 group/worker generation；失效时返回 8408，frontend 丢弃展开结果并最多重选一次。ViT-only 取消或响应丢失不能取消同请求号的 P/D admission。
5. P 调用同一个 ViT 获取 embedding，重新计算行 hash，并校验 layout 长度和 frontend token_ids。D 沿用 P/D 流程。BATCH dispatcher 必须把已选择的 ViT 地址与 P/D 一并下发。

## 启动与连接

```bash
java -cp flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar \
  org.flexlb.mockengine.MockVitServer \
  --port 62000 --hidden-size 128 --concurrency 4 --queue-capacity 64 \
  --embedding-cache-bytes 67108864 --hash-cache-bytes 8388608 \
  --compute-delay-ms 1
```

每个 ViT worker 使用独立监听地址，Master 的 `MODEL_SERVICE_CONFIG.role_endpoints` 增加 `vit_endpoint` 并在服务发现中提供 HTTP 端口（gRPC=HTTP+1）。frontend 使用 remote ViT 配置，并传递模型的 multimodal separator 配置。P/D 和 frontend 的 KV block size 必须一致；要模拟原生 token KV key，P/D 使用 whale 模式，可用 `--whale-bundle true` 在独立测试进程内运行。各角色 JAR 应先复制到运行目录，再启动，防止重新打包覆盖正在运行的产物。

ViT 接受 base64 data URL 和 HTTP(S) 图片；按 Qwen2-VL 的 28 像素粒度、min/max pixels、显式宽高计算 token 数。embedding 为 `[rows, hidden_size]` FP32，行 hash 与 main 的 `FeatureHash.h` CPU 算法一致。URL cache key 对齐 Linux libstdc++ `std::hash` 和预处理配置；相同图片重复出现时保留多个输出片段，共享计算。

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

`MOCK_VIT_ENABLED=1` 在 bundle 中启动独立 ViT JVM，并与 Master、P/D 一起监管退出。默认使用最后一个 P/D gRPC 端口的下一个端口；可用 `MOCK_VIT_PORT` 显式设置。其他参数对应 `MOCK_VIT_HIDDEN_SIZE`、`MOCK_VIT_CONCURRENCY`、`MOCK_VIT_QUEUE_CAPACITY`、`MOCK_VIT_EMBEDDING_CACHE_BYTES`、`MOCK_VIT_HASH_CACHE_BYTES`、`MOCK_VIT_COMPUTE_DELAY_MS`。Master 的组合服务发现文件保留 P/D 动态更新并加入 ViT；原始 P/D 文件仍由 Mock 独占写入。`identity.json` 记录 ViT 参数，日志在 `vit.log`。旧 Master 模式拒绝启用现代 ViT。

### 隔离 frontend 测试镜像

`whale_frontend_fixture.py` 是通用图片协议的 CPU 验收入口，要求显式设置 `RTP_LLM_MOCK_FRONTEND=1` 和 `MOCK_FRONTEND_SOURCE_ROOT`。它只使用所声明的打包源码及 protobuf，不会下载旧版 frontend。原生 `libth_transformer_config.so` 必须与打包源码兼容；使用原生启动 supervisor。

该入口在运行目录生成小型 Qwen2-VL 模型配置和确定性 tokenizer，默认 HTTP 端口 20550、KV block size 64，继承部署的 `MODEL_SERVICE_CONFIG` 连接真实 Master。它只生成模型元数据，不加载真实权重。配套 Master/P/D 必须同样使用 block size 64；此配置用于协议和缓存正确性验证，不能沿用原部署的容量或性能结论。

多 worker 验收应先自然预热，再检查热前缀；大量空闲 P 的轮转可能使少量重复请求仍落到冷 worker。应同时核对 `prefix_len`、P 的 `cache_key_hits`、D 的 `decode_reuse_blocks` 和完成后的请求/响应缓冲清理，不能只以 HTTP 200 代表缓存验收通过。

## main V4.1 prepared images

独立 ViT 增加 `--model-path /path/to/checkpoint`；bundle 使用 `MOCK_VIT_MODEL_PATH`。
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

## 集成提交边界

本分支先合并 next + main，复用 main 原生 frontend/Master/协议。Mock 功能提交只修改 Mock engine 与 tools/mock_bundle。
单独的生产修复提交分别保留 BATCH 的 ViT 地址、对齐 frontend 与 CompleteTokenIds 的图片身份编码，以及允许 prepared V4.1 不带 URL affinity keys 选择 ViT；这些修复与 Mock 功能独立 review。
旧分支的 Whale 验收记录只对应旧源码，不作为这份集成分支的运行验收。

协议集成 smoke 使用 `vit_protocol_smoke.py --master host:grpc-port --control http://host:mock-control-port --processor-identity <sha256> --output result.json`。
它使用生成的 protobuf，对真实 Master 调度、Mock ViT、P/D 执行及排空做检查；必须分别启动 NON_BATCH/BATCH 环境。
覆盖 URL 图片、V4.1 prepared 图片、text-only prepared metadata。此客户端不覆盖真实 frontend 的 HTTP 入口、原生扩展加载或真实图片预处理。
