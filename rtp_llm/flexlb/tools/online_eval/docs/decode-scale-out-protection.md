# Decode 扩容保护

实例名为 `elastic_lifecycle::decode_scale_out_protection::<profile>`，覆盖四种 profile。流程在 `case_programs/elastic_lifecycle.py`，参数在 `scenarios/elastic/lifecycle.yaml`，沿用已有弹性程序，没有新增单 case Python 程序。

默认从 2P/2D 扩到 2P/3D，每个 Decode 的 `decode_max_engine_requests` 为 8。客户端保持最多 24 个未结束请求，输入 512 tokens、输出 128 tokens；每次请求使用独立 KV key。先确认每个旧 Decode 已完成请求且当前负载至少为容量的一半，再添加新 Decode。

“新 Decode 不能打死”的验收条件是：

- 从扩容前开始采样，新实例不能停止、消失、重置计数，采样中的 running + waiting 和 Master `engine_capacity_used` 均不能超过配置上限。
- 新实例不能发生 KV admission 失败，KV 可用块和引用块计数不能越界。
- 新实例在扩容后的四个连续观测窗口内都必须完成请求；旧实例在扩容后也要继续完成请求。不给新实例分流不能算通过。
- 停流后每个已发请求都必须业务成功且消费线程结束，Master 的所有权计数归零，新 Decode 空闲且释放 KV 引用。

默认观测 20 秒，每约 100ms 采样一次；HTTP 开销计入实际间隔，间隔超过 1 秒视为证据缺口。原始样本、请求账本和判定分别保存到 `decode-scale-out-observations.json`、`decode-scale-out-requests.json`、`decode-scale-out-verdict.json`，瞬时越界不会被最终恢复掩盖。

YAML 可调整 PD 数量、容量、并发数、输出长度和观测时间；并发必须大于旧 Decode 池的总容量。Python 固定执行顺序与断言。容量检查使用实际 dispatch 占位和引擎负载，不把 Master 排队请求数当作引擎执行容量。

这项测试使用 Java Mock，验证调度与完成性；采样不能证明两个采样点之间绝无短暂峰值，也不等同于真实 GPU 的显存 OOM 验证。

运行示例（在已取得租约的远端容器中）：

```bash
python3 parallel_runner.py --parallel 1 --profile batch-window \
  --instances elastic_lifecycle::decode_scale_out_protection::batch-window \
  --out-dir /tmp/decode-scale-out
```
