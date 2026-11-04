# 分布式 Engram：代码 review 入口

当前方案：RTP worker 在 P 个 host 上各持有一片权重，一个或多个实例共享这些分片。
`engram_sidecar` 只是保留的目录名，当前没有独立 sidecar 或集中式整表 owner。

建议按提交顺序 review：

1. **存储与模型接入**：`engram_shard*`、`engram_manifest`、`engram_checkpoint`、
   `engram_publish`、`engram_storage`、`engram_bootstrap`、`engram_model_attach`；
   `engram.py` / `deepseek_v41_model.py` 的接入，以及对应核心 UT。
2. **启动与交换**：`engram_startup`、`engram_exchange`、`BackendManager` 钩子；
   单 gang 复用 Store，跨 gang 临时 RPC；对应启动、超时和 mTLS 测试。
3. **验证工具**：真实 TCPStore 测试、CUDA 原语和分片生命周期测试、跨 host 合成探针。
4. **使用说明与 review 记录**：本文件、[配置契约](framework/STATIC_RPC.md)、[本轮清理说明](REVIEW.md)。

## 运行测试

```bash
# 标准库测试，不加载 CUDA；PYTHON 可指定解释器。
bash tools/engram_sidecar/test.sh --cpu
# 在服务的 Torch/safetensors 环境运行，包含实际文件读取和 TCPStore。
bash tools/engram_sidecar/test.sh --torch
```

实际 CUDA 测试需可用的 HOST_NUMA/FABRIC/IMEX 环境和预编译 native bridge，见
[native/README.md](native/README.md)。当前 bridge 仍是开发扩展，并未在这轮整理中改造成正式 Bazel wheel 依赖。

## 历史资料

重写前完整分支及所有原始结果已保存在
[`codex/engram-archive-20261104`](https://github.com/MMadhatter/rtp-llm/tree/codex/engram-archive-20261104/tools/engram_sidecar)。
旧单 owner 工具、一次性 benchmark 数据、迁移过程记录不再混入当前 review diff。
这些资料仍可复现旧实验，但其结果不能当成最新分布式版本的验收证据。
