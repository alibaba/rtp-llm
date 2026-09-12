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
