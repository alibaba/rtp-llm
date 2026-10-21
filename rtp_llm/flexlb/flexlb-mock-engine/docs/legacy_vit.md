# dsv41_ced_release Mock ViT

此适配基于 `develop/dsv41_ced_release`，用于验证旧图片请求链路。它不合入现代 ViT 分支；真实 GPU 计算、RDMA 与线上事故机制仍需独立复现。

## 协议和执行路径

旧版 `MultimodalInputsPB` 的字段 2 是 `metadata_only`、字段 3 是 V4.1 prepared inputs，返回逐图 `MultimodalOutputsPB`。旧 Master 的 `vit_only` 位于 Schedule 字段 16。main 已使用不同的字段类型与返回结构，不能共用端口自动猜版本。

普通图片按以下顺序执行：frontend 向 Master 要 ViT，调用旧 metadata-only RPC 得到每图 token IDs，展开后向 Master 申请 P/D；下发给 P 的 GenerateInputPB 仍保留原始图片占位符和选定 ViT 地址。Mock P 通过旧 feature-only RPC 取 BF16/FP32 embedding，计算相同的 CPU 行 hash 并展开占位符，随后按完整 token 数进入原有 P/D 流程。`MOCK_VIT_IMAGE_TOKEN_ID` 必须设为模型实际图片占位符 ID，不能使用测试值替代线上配置。

V4.1 的 token IDs 已由 frontend 准备，不调用 metadata-only RPC。Mock P 校验 ViT 返回的片段行数，并保留 canonical IDs。图片 cache keys 对齐 release `HashUtil.h`、`CompleteTokenIds::imageCacheIdentity` 和 `KVCacheHashUtil.cc`：纳入图片内容 SHA、processor identity、几何和 span；可选 bounded replay seed 从 `DSV41_CED` / `DSV41_SWA_BOUNDED_REPLAY` 读取。复用长度不能切入图片中间，也不能复用最后一个 prompt token。这里模拟 key/复用边界，不模拟 CED 的完整 GPU 执行和 KV 数据。

P 上的 ViT 获取有独立的有界执行队列，避免在旧 Mock 的 Netty direct executor 上等待图片 RPC。纯文本与显式 online_eval replay key 路径沿用原有行为。

## CPU ViT 进程

```bash
python3 -m rtp_llm.server.mock_vit_engine \
  --port 62000 --hidden-size 128 --concurrency 4 --prepared-v41
```

需要 Torch、PIL、生成的 release protobuf，以及与 release Python 源码匹配的 `libth_transformer_config.so`。启动环境应使用本分支源码，不能混用 main 的 protobuf。hidden size、processor 配置、dtype 应与被测模型一致；CPU 后端当前使用 BF16。

实现复用 release 的 `MultimodalRpcServer`，保留 active request/byte admission、状态 epoch 微秒版本推进、token-ID 数量/时间窗口缓存、V4.1 串行 forward/export 锁与异常清理。仅替换 forward，按 worker 注入 CPU hash 函数；真实服务默认 hash kernel 不变。

URL fixture 目前只接受 base64 data 图片，使用 Qwen2-VL 28 像素粒度计算长度，不能视为 DeepSeek-V4 原始图片 processor 的精确 token 数。V4.1 输入则使用真实 release processor 的 prepared patches、types 和 identity 校验，按 types 长度生成确定性特征。GPU 性能、内容审查、RDMA slot/lease 与动态 batching 不在此 CPU 后端的验证范围内；HTTP frontend 的独立 Pod 验证见下文。

## 验证

`mock_vit_engine_test.py` 使用真实旧版 gRPC server 和 frontend `_get_vit_token_ids` 方法，覆盖 metadata/feature 两次获取、独立 token cache、V4.1 packed 校验、状态、并发、超时与 hash 一致性。`LegacyMockVitClientTest` 验证 P 的逐图响应消费和 canonical IDs；`LegacyImageCacheKeysTest` 使用 Linux C++ 参考值校验图片身份与两种 seed。

```bash
python3 -m rtp_llm.server.legacy_vit_wire_smoke \
  --master-grpc 127.0.0.1:7003 --output /path/to/run/legacy-vit-wire.json
```

协议探针执行真实旧 frontend metadata client → Master → Mock P → ViT → Mock D，分别验证 NON_BATCH/GenerateStreamCall 与 BATCH/FetchResponse。普通图片输入 66 token、热前缀 64；prepared V4.1 输入 129 token、完整图片热前缀 128。它不启动 HTTP frontend，也不证明事故已复现。

远端验证曾使用旧容器的 native config 库；该库缺少新 seed 常量。仅测试启动脚本临时提供了 `CacheKeyConfig.h` 中的准确常量，未替换 native KV hash 实现，也未把此脚本写入产品代码。这仅是早期协议探针的环境记录；随后已在匹配 release 的 Whale 镜像中重验真实 native seed/hash 和完整 HTTP frontend，未替换 native 常量。跨 134/111 的 Master 连 ViT 使用测试参数 `-Dflexlb.engine-grpc.connect-timeout-ms=1000`；默认 20ms 的跨主机尝试未通过，源码默认值保留。这些差异不能作为线上事故因果证据。

## 独立 Whale 部署

release 分支为 `codex/mock-vit-dsv41-release`。frontend 直接运行
`/opt/conda310/bin/python3 /opt/mock-frontend/src/rtp_llm/flexlb/tools/mock_bundle/whale_frontend_fixture.py`，
不通过 `bash -c` 拼接参数。平台 checkpoint 为空时，启动器读取真实 tokenizer
资源中的模型元数据；只下载配置、tokenizer 和 encoding，不下载权重。
ViT 使用同一入口，设置 `MOCK_BUNDLE_ROLE=vit`；Java Master 和 P/D Mock
仍由 `sh /home/admin/whale_start.sh` 按角色启动。四种角色位于独立 Pod。

该 V1 模板会用根镜像覆盖 inference/ViT 的资源模板镜像，因此根镜像必须同时
包含匹配 release 的 Python/native 库、Java 21 与已验证的 Master/Mock JAR。
CI 的 `frontend_bundle_image` 参数将指定 bundle 复制进 Python 运行镜像，并
复核 bundle 清单中的 JAR SHA256；该参数和源码 SHA 一起纳入构建 identity。
部署后必须读 Carbon 实际镜像与进程，不能只检查模板 `package_infos`。

历史窗口中的 P/D 各有 8 个 KV pool。当前 Mock 的单 pool 配置
P=45083、D=932802 只是选定的实验容量，不等价于真实多 pool allocator；
也不代表已复现 RDMA 或 CED 事故。历史拓扑为 2 Master、45 frontend、20 ViT、
40 P、160 D；先以较小 frontend/ViT 数量验通协议，再独立扩大规模。

## Whale HTTP 验收

```bash
python3 -m rtp_llm.server.legacy_vit_http_smoke \
  --frontend http://<frontend-ip>:20550 \
  --control http://<mock-ip>:29129 \
  --expected-workers 200 --output /path/to/run/release-http.json
```

此探针只依赖 Python 标准库，生成确定性 PNG。它执行真实 HTTP renderer 和
chat completion，覆盖文本、224x224 单图、1024x768 单图、多图、重复图片、
单 token 输出、SSE 流式输出、坏图片后的文本恢复；核对 renderer/input usage、
输出 token 数、选中的 P/D/ViT 地址，以及 Mock running/waiting/held/referenced 排空。
坏图片在该 release 的 HTTP frontend 返回 500，探针只将它作为预期拒绝路径；
不能把它计为成功图片请求。

2026-10-21，专用部署 `na130_bailian_cpu_mock_cache_repro` 的 biz 16 / 模板 25
完成发布：2 Master、10 frontend、1 ViT、1 独立 Mock Pod（40 P、160 D）。
Master 为同一 ZK role 下的一主一从；独立启动器使用真实 process_ok/after_start
生命周期 hook，重启后自动上线。23 项 Java 回归、8 项真实 native 环境中的
ViT RPC 测试、8 项启动器测试通过；10 个新 frontend 各执行文本和图片请求，
20 次均为 HTTP 200。完整探针输入 token 为文本 35、小图 219、大图 531、
多图 716，均与 renderer 一致；输出 4 和 1 token 的计数也一致。
ViT 活跃任务为 0，Mock 的运行、等待、持有块与引用块均归零。

最终镜像 CI 为 81753260，组合的 Master/Mock bundle CI 为 81750798。
frontend/ViT 与 Mock 源码为 `0706104ba6`，Master Java 源码为 `1b81cc980d`；
其 Java 实现未在这些启动器/Mock 修复中改变。该结果验收了 release 协议，
不代表历史 45 frontend / 20 ViT 的规模验证、8-pool 容量等价或线上事故复现。
