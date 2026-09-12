# Whale 寄生 mock 模式

显式设置 `RTP_LLM_MOCK_BUNDLE=1`、`FETCH_OUTPUT_STREAM=0` 后启用。
一个 CPU Pod 运行 master 与 mock 两个独立 JVM；mock 内每个逻辑引擎拥有独立端口、KV 池、队列和生命周期。
默认的本地 case 模式和 Whale 单引擎 Pod 模式不变。

`bundle.yaml` 定义 P/D 数量、每引擎容量、线程数、堆大小与观测参数。
默认 48P/192D，P=11042 块、D=27686 块，每块 1024 token，均为单池。
性能文件不修改输入或输出长度；现有 `FLEXLB_CONFIG` 优先传给 master 与 mock，避免两份估算配置漂移。

master 通过本地 discovery 文件发现各引擎，不依赖 P/D VIP。
控制端口使用 Pod IP；引擎 RPC 使用独立 loopback IP 和端口，保证 master 的 engineIp 指标不互相覆盖。同 Pod P→D 仍使用现有 RPC 协议。
框架自动接续，不等待客户端 Fetch；启动任何对端失败时 supervisor 会关闭另一 JVM。

指标保留真实 Pod 的 `hippo_role` 和 `container_ip`，用 `engine`、`engine_port`、`dp_rank` 区分逻辑引擎。
每个引擎的累计计数、时间基准和执行轮采样单独维护；不能把共享 Pod 的总数当作单引擎值。
现有 Grafana 查询应选择 master 角色，并按 engine 或 dp_rank 查看逻辑实例，不能要求出现不存在的 P/D Pod。
不修改 Grafana 面板。

## 验收边界

分别核对调度接受 QPS、decode 完成 QPS、失败数、输出 token 数、Fetch RPC=0、队列与 KV 归零。
调度确认不代表推理完成。现有复制流量的 gRPC 入口通过显式 `RTP_LLM_MOCK_SCHEDULE_ONLY=1` 开关调用测试侧的 schedule_only.py；返回带 schedule_accepted=true、inference_completed=false 的确认帧，不输出 token 或推理完成标志。未开启时走原推理路径。
当前 max_new_tokens 被模拟器视为实际输出长度，缺少 EOS 模型；超大上限请求仍可能长期占用资源。
未完成真实复制流量验证之前，不宣称成功率或性能已对齐。

Bundle jars use the explicit Maven profile `opensource,!internal,whale-bundle`: KMonitor is included, while engine discovery remains local to the Pod. VipServer is intentionally absent from these test jars; the default internal profile is unchanged.


### 可选的提前 EOS 模型（默认关闭）

真实引擎把 `max_new_tokens` 当上限，在 `min_new_tokens` 后允许 EOS；
`ignore_eos=true` 禁止 EOS。测试 mock 可在 performance JSON 中配置：

```json
{"decode":{"eos":{"enabled":true,"distribution":"geometric","mean_tokens":1024,"seed":20260912}}}
```

Whale bundle 也可通过 `MOCK_EOS_CONFIG_JSON` 传入上述 `eos` 对象。
不设置或 `enabled=false` 完整保留旧行为；仓库默认配置不启用。
启动时生成运行目录下的 performance.json，不修改原始流量及 master 可见参数。

这是常量 EOS 概率模型：允许 EOS 后，每 token 的结束概率为 `1/mean_tokens`。
默认最小长度为 1 时，未被请求上限截断的平均长度是 `mean_tokens`；
较大的 min_new_tokens 会推迟分布起点。请求 ID 和 seed 决定采样，P/D 与重试一致，
不会随并发或执行步数重新抽样。显式 replay output_len 优先，启用模型时仍遵守请求上下限。
完成沿用现有 decode 正常收尾路径，释放 KV/执行位，无需 Fetch。

示例 1024 是暂定模拟参数，**没有当前生产输出分布校准，不代表生产均值**。
此模型对齐结束约束，不模拟语言内容、真实 token EOS、stop words 或 beam search。
真实分布不是几何分布时，应使用真实 replay output_len 或重新校准模型，不能靠调短长度证明生产吞吐。

验证使用实际 D 完成计数及 `mock_generate_tokens_total` 增量，输入侧用
`mock_context_tokens_total` / `mock_context_compute_tokens_total`；这些累计计数不受抓取窗口重置影响。
前端 schedule acknowledgement 仍不是推理完成，不能当端到端成功率。
