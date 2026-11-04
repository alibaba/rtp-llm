# Review 范围与验证记录（2026-11-04）

基线：`feature/dsv41-release-20261019` 的 `c78d885fd504910b301e987aca199f788249c3b3`。
分支：`codex/engram-sidecar`。整理后的四个提交依次为存储/模型、启动/交换、验证工具、文档。
每个功能提交携带对应 UT；不要把后续工具提交误认为运行时新增依赖。

## 清理内容

- 旧单点 owner 加载器 `LoadedGeneration/load_checkpoint` 及专属 UT 已移除。
  `engram_checkpoint` 保留元数据、文件一致性检查和有界原始读取；实际分配/分片加载统一由 `engram_publish` 完成。
- 删除旧独立 `CapturedEngramGraph` 包装器及仅用于该包装器的 mock Graph 测试。
  RTP 服务使用的 process-lifetime pin、alias、失败回收及 Graph 前置检查仍有测试。
- 删除同一“没有 exchange package”条件的重复 UT、空 release 等仅复述实现的用例。
- 删除旧 owner/consumer 部署工具、其测试及一次性实验产物在本分支的副本。
  全量资料位于归档分支，未丢弃原始证据。
- 保留当前分布式链路的核心 UT、真实文件读取、TCPStore、native 原语、GPU 分片生命周期和跨 host 合成探针。

本次 review 同时修正两处问题：

1. 旧 inspector 按“完整表装入单机”的 allocation 预算限制模型。现在先检查元数据，
   再按实际 P 分片，由 publisher 检查本 host 的 padded allocation/budget，避免错误拒绝合法分布式分配。
2. 发布结束前重新检查 checkpoint 索引快照；加载期间索引变化会失败并回收本轮分配，不返回可用 descriptor。

新测试直接调用当前 `load_rows/publish_shard`，检查 FP8/scale 字节、切片范围、分块上限、摘要和异常回收。

## 验证

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
- external async prefetch 尚未启用，Graph 仅开放已限定的 decode/process-lifetime 路径。
- native bridge 仍需预编译并放入运行环境；正式 wheel/镜像依赖集成需要独立完成。

历史完整版本：
[归档分支](https://github.com/MMadhatter/rtp-llm/tree/codex/engram-archive-20261104)。
归档起点为 `3f4e0f6a1943c232e1ed8185e2c2cbf1133c2bcf`，包含旧提交历史及所有性能原始结果。
