# Review 范围与验证记录

当前审查基线为 `1790c86bb`，范围是 `serina.wzq@alibaba-inc.com` 的三个提交：

1. `8b32ccf74`：存储与模型接入。
2. `6d7828b5d`：启动与交换。
3. `29474832a`：验证工具与文档。

本轮修复分支：`codex/engram-cleanup-review-20261106`。修复保留 dev-main 的量化 lookup、
同节点 scale shards、decode schedule 以及 Graph pin/view lease/event drain 约束。
外部表支持 decode scheduled prefetch，通用 prefill 异步 lookup 仍不支持外部表；
两者分别使用 `supports_decode_prefetch` / `supports_generic_prefetch` 表达。

## 2026-11-06 修复结果

- 普通 embedding 调用固定返回 BF16，显式量化入口保留；统一 lookup guard、事件注册和关闭状态，保留 Graph pin、view lease 与 enqueue 失败后的读取保护。
- 外部 scale 直接复制当前 rank 的片段，cache hit 不读取整表，singleton 保证独立 GPU 副本。现有 scale helper 仅新增显式 `external_source=True` 适配，默认 CPU 入口及原 collective/cache 顺序不变。
- 本机 publisher 预算与 consumer 全组映射上限分开；全组硬上限按实际 host 数计算。manifest 只解释一次，预先完成所有层的 AttachPlan，复用已验证的完整模型布局。
- native 与 sharded mapping 共用管理、状态报告和失败回收；删除无调用方的 descriptor 文件入口。补齐 Bazel ganginfo 依赖、standalone contract runner 和真实 Fabric lookup/Graph 回归。

## 2026-11-06 实际验证

在个人开发容器的独立目录 `engram-cleanup-20261106` 验证，未构建镜像或修改线上部署。

| 验证 | 结果 |
| --- | --- |
| 本机标准库 CPU | 171 项通过 |
| Linux 服务环境 `test.sh --torch` | 231 项 + 20 项 standalone embedding contract 通过 |
| schedule CPU/CUDA 与原 scale CPU 回归 | 27 项通过 |
| 真实 P=2 Fabric lookup / 变 ID Graph / pin 与 alias 回收 | mapped scales、默认 GPU scales 两种配置通过 |
| 双 GPU external scale + scheduled decode + Graph | 两个 rank 均通过，使用真实 scale group |
| 更新后的 CPU warmup harness | Linux 5 项通过，本机相关 22 项通过 |
| exchange 声明 runfiles 闭包 | 40 项通过；未执行实际 Bazel 构建 |
| P/D off → on → off | 每轮 26 条矩阵请求 + 8 条 logprob 请求通过，覆盖全部 8 个 decode 实例 |

端到端使用两台 CP4 prefill、两台 EP8 decode。三轮完整 token 均与修复前的 26 条请求基线一致，
三轮补充请求的 top-5 logprobs 完全一致。覆盖短问答、长上下文、串行和并发，SSE 收集的 token 数与 usage 一致；
请求返回的 DSpARK 验证 token 数等于 `5 × draft_rounds`。

运行日志确认全部 decode rank 的 batch 1/2/4/8 Graph capture 成功；Engram layer 1/14 的 prefetch 已准备，
每节点 `scale_shards=4`，开关表分片时分别走 external / local 表。最终保持 decode shard 关闭，prefill P=2，
CUDA Graph、DSpARK gamma=5 和 decode prefetch 开启。

完整文件读取回归在 Linux 服务环境执行。本机 macOS 的完整 Torch 入口有两项 checkpoint 测试依赖 Linux
`/proc/self/fd` 而失败；未修改该平台契约。CPU warmup 对 Triton 的无意依赖已在测试边界隔离，并在两端复验。

## 2026-11-06 平台 gang 注解发现验证

两台 prefill 各自使用 `WORLD_SIZE=4`、启动 `WORLD_RANK=0`、`LOCAL_WORLD_SIZE=4`、
`TP_SIZE=4`、`PREFILL_CP_SIZE=4`、`DP_SIZE=1`、`EP_SIZE=4`，共享 `ENGRAM_SHARD_COUNT=2`。
两台分别保持独立计算 world 和 TCPStore，不合并为 WORLD_SIZE=8。

个人开发容器提供平台同格式的 `app.c2.io/biz-detail-ganginfo` 注解文件，由
`GANG_ANNOCATION_PATH` 指定路径；移除显式 cohort、instance index、ranks per instance 和 ganginfo 目录配置。
这是两台真实 GPU host 上的注解消费与交换验证，没有新建 C2 调度部署。

- 八个 prefill worker 均记录 `source=annotation`、`single_gang=False`，两台自动匹配 index 0/1，
  P=2 发布、跨 host 交换及导入成功。两台启动交换端口在 READY 后均无监听。
- 新增集成测试使用两个真实 TCPStore、八个独立客户端和真实启动 HTTP 交换，覆盖注解发现、
  两次分片发布、八个相同 package、Store 清理和监听关闭；分片 descriptor 使用测试替身。
- Linux `test.sh --torch` 更新为 **232 项 + 20 项 standalone contract 全通过**。
- 新跑 26 条矩阵请求和 8 条 top-5 logprob 请求，覆盖全部八个 decode 实例；完整 token 与原基线一致，
  补充请求 token 和 logprobs 与此前 off/on/off 结果一致，差异数均为零。
- 最终 10/10 服务端点健康。decode 未重启，保持表分片关闭、EP8、CUDA Graph、DSpARK gamma=5、
  Engram prefetch 和原 scale shards 优化开启。

该拓扑复用现有启动实现，本次只新增回归测试和配置说明。

## 2026-11-06 回归入口

- `test.sh --cpu`：标准库存储、发布与启动合约。
- `test.sh --torch`：再加入真实文件读取、TCPStore、external scale source 及 standalone embedding 合约。
  最后一项通过其 `main()` 初始化；任意阶段失败均使总入口失败。
- `//rtp_llm/models_py/modules/dsv41/test:test_engram_exchange` 的窄依赖包含 `engram_ganginfo`；
  embedding 合约另有 `//tools/engram_sidecar/native:test_embedding_contract` CPU target。
- `ENGRAM_RUN_CUDA_TESTS=1 bash tools/engram_sidecar/test.sh --cuda-lookup`：持久化的小 P=2
  真实 Fabric mapping → `from_storage` → 生产 BF16 lookup / 变 ID Graph 回放。
  覆盖 mapped scales 和默认 Blackwell GPU scales、int32/int64、非 24 列、跨片边界、pin/alias 回收。
  两个 owner 位于同一测试机；跨 host 模型、attention schedule、DSpARK 和请求输出仍由 P/D E2E 验证。

本轮实际执行结果见上表；下列历史通过项不代表本轮新代码的测试结果。Bazel 镜像构建不在本轮范围内。

## 2026-11-04 历史清理记录

以下记录对应当时的 `codex/engram-sidecar`，基线为
`c78d885fd504910b301e987aca199f788249c3b3`，当时拆为四个提交；它不是当前三提交的基线。


- 旧单点 owner 加载器 `LoadedGeneration/load_checkpoint` 及专属 UT 已移除。
  `engram_checkpoint` 保留元数据、文件一致性检查和有界原始读取；实际分配/分片加载统一由 `engram_publish` 完成。
- 删除旧独立 `CapturedEngramGraph` 包装器及仅用于该包装器的 mock Graph 测试。
  RTP 服务使用的 process-lifetime pin、alias、失败回收及 Graph 前置检查仍有测试。
- 删除同一“没有 exchange package”条件的重复 UT、空 release 等仅复述实现的用例。
- 删除旧 owner/consumer 部署工具、其测试及一次性实验产物在本分支的副本。
  全量资料位于归档分支，未丢弃原始证据。
- 保留当前分布式链路的核心 UT、真实文件读取、TCPStore、native 原语、GPU 分片生命周期和跨 host 合成探针。

当时的 review 同时修正两处问题：

1. 旧 inspector 按“完整表装入单机”的 allocation 预算限制模型。现在先检查元数据，
   再按实际 P 分片，由 publisher 检查本 host 的 padded allocation/budget，避免错误拒绝合法分布式分配。
2. 发布结束前重新检查 checkpoint 索引快照；加载期间索引变化会失败并回收本轮分配，不返回可用 descriptor。

当时新增的测试直接调用 `load_rows/publish_shard`，检查 FP8/scale 字节、切片范围、分块上限、摘要和异常回收。

## 2026-11-04 历史验证

- 本机标准库核心测试：155 项通过。
- GB200 a0004 个人容器隔离目录 `/tmp/engram-review-20261104`：211 项通过，包含核心 UT、
  真实 Torch/safetensors 文件检查、bootstrap、模型加载/warmup 合约和真实 TCPStore。
- `git diff --check` 通过；重组提交后必须校验最终 tree 与已验证快照一致。

运行入口：

```bash
bash tools/engram_sidecar/test.sh --cpu
bash tools/engram_sidecar/test.sh --torch
```

没有重启线上/实验 P/D，没有重新进行全模型 chat、跨 host GPU、Graph 性能或 <100us decode 增量验收。
Bazel 依赖声明随相应提交整理，本轮未运行完整 Bazel 镜像构建。

## Review 时仍需关注

- checkpoint 索引哈希不是完整权重版本证明；同组必须使用不可变、相同模型。
- 启动轮次在内部隔离；运行中 host 退出后的故障恢复仍要求整组重建，不提供热替换或在线 reshard。
- 跨 gang 当前每 host 一条启动连接；多 host 实例尚未合并成每 gang 一条。在线没有 RPC。
- external decode scheduled prefetch 已支持；generic prefill 异步 lookup 不支持外部表。Graph 仅开放已限定的 decode/process-lifetime 路径。
- native bridge 仍需预编译并放入运行环境；正式 wheel/镜像依赖集成需要独立完成。

历史完整版本：
[归档分支](https://github.com/MMadhatter/rtp-llm/tree/codex/engram-archive-20261104)。
归档起点为 `3f4e0f6a1943c232e1ed8185e2c2cbf1133c2bcf`，包含旧提交历史及所有性能原始结果。
