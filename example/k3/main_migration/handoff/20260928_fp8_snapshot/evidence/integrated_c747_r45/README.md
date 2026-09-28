# 四层 FP8 + MTP PD r45 功能记录

源码固定在 `c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2`，111 负责 Prefill，112 负责 Decode，均为 TP8/EP8。四层 target 经 FastSafetensors 从 `/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers` 直接读取；MTP checkpoint 视图位于各机个人数据盘，权重 shard 指向 3FS。两端均使用任务进程内的 64 线程预读库。两端各八个 rank 的启动日志通过 FastSafetensors 检查，没有 loader 回退。

`r45-flow-prefill-111-evidence.tar.gz` 含 11 条原始请求、运行器结果、独立审计、Prefill 启动日志和权重预检。`r45-decode-112-startup-evidence.tar.gz` 含 Decode 启动日志和权重预检。`fleet-selection.json` 是旧任务服务退出后、r45 启动前的全机 GPU 筛选快照；111/112 当时均可独占使用，双机 RDMA HCA 均为 ACTIVE。

11 条 flow 请求全部在 300 秒内返回。独立审计逐条复查 HTTP 200、PD 交接长度、Decode 路由和 MTP draft 轮次，结果为 `passed=true`。这组四层权重只检验链路；审计在随机模型输出里记录了 2 个替代字符，**不支持答案语义正确性结论**。这次运行仍使用 256 token 的旧 flow 上限；后续脚本已改为 16 token，减少重复功能检查耗时。

111 首次启动时端口预检遇到短暂的 `25165` 占用，未加载模型；失败现场留在 111 的 `k3prefill-3fs-111-r45-c7479de-port-conflict*`。重试时 3FS 权重结构检查出现约两分钟延迟，随后预检通过，真实 FastSafetensors 读阶段继续完成。尚无证据将这次延迟归因于挂载实例；不能把它当成持续的权重读取吞吐。

归档 SHA256：

- `r45-flow-prefill-111-evidence.tar.gz`: `6cbc5b5ce649e7c8cc29a343bbf3c607b16400766d980792183d57112aca126f`
- `r45-decode-112-startup-evidence.tar.gz`: `20095fe4941176e0828bac238ac8daf484c134b5bcfe25e8c940b397cb623e95`
