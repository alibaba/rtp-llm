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

目前仅支持 URL IMAGE 与 inline gRPC FP32。尚未模拟视频、音频、预处理 tensor、V4.1 packed patches、模型特定位置编码、GPU 缓存 tier、RDMA 租约或真实内容审查；packed 输入会显式拒绝，不能被当作纯文本成功处理。GreenNet 接口仅代表真实实现的 disabled/noop provider。固定延迟和线程池不代表真实动态 batching 性能。

`develop/dsv41_ced_release` 的 `metadata_only` 与 main 的 `support_rdma` 占用同一字段号，旧版返回逐图 `MultimodalOutputsPB`，不能在同一监听地址自动猜测协议。旧路径适配和线上事故复现应在独立分支、独立进程中验证；现代通用图片链路通过不等于 V4.1 事故已复现。
