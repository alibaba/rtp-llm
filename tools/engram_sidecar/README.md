# Engram 跨 host 分布式存储开发工具

当前方案是 **P 个存储 host 分片持有 Engram，一个或多个 RTP 实例映射同一组分片**。P 与实例数、TP/DP/EP 独立。`engram_sidecar` 仅为历史目录名；已取消 sidecar 部署和完整权重集中在单个 owner 的方式。

- [一次性 RPC：设计、启动配置、接口与真实跨机验证](framework/STATIC_RPC.md)
- [N 个实例静态共享语义与地址配置](framework/STATIC_INSTANCE_GROUP.md)
- [当前审查、使用方法、边界和测试记录](framework/SHARDING_REVIEW_20261030.md)
- `rtp_llm/models_py/modules/dsv41/engram_publish.py`：每个实例在启动时加载自己的 weight/scale 行分片，持有 CUDA HOST_NUMA/FABRIC allocation，并在内存里生成 descriptor（不落文件、无独立 holder 进程）。
- `rtp_llm/models_py/modules/dsv41/engram_bootstrap.py`：实例启动时校验 P、导入映射并校验每个 shard 的边界样本。
- `rtp_llm/models_py/modules/dsv41/engram_shard_storage.py`：将所有分片拼接为连续 weight/scale 虚拟地址；在线 kernel 直接读内存，无查表 RPC。
- `rtp_llm/models_py/modules/dsv41/test/`：CPU 测试；`test_engram_descriptors` 覆盖 descriptor 的几何/身份/handle 唯一性负例，`test_engram_publish` 覆盖分片布局与放置补齐。
- `native/check_shard_mapping.py`：真实 CUDA mapping 基础探针；单块/同 host 测试用于验证原语，不表示单点部署仍受支持。
- `framework/sharded_pd.py`：保留的四节点 PD 实验脚本，不是通用生产部署入口。

历史性能报告与 `results/` 保留供追溯，其中旧 owner/sidecar 部署命令不代表当前方案。当前支持静态组启动 RPC 自动交换 handle，保留离线文件诊断模式；未实现动态发现、在线 reshard 或高可用。持片已并入模型进程，因此没有独立 holder 生命周期；分配随进程退出回收，`engram_publish.release()` 供探针和重启路径使用。
