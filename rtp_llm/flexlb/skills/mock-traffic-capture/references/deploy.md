# 部署与各角色配置

在 Whale 核对部署 ID、Zone、角色和当前镜像。按现有模板及 Zone 配置发布镜像和环境变量，用 `whale-biz-operation` 完成正常授权和发布。采集前从运行 Pod 回读实际镜像、私网连通性、进程参数、实例数和工作盘，保留既有 P/D、Master 和 ViT 的发现配置。

| 角色                                | 采集配置                                                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Python Frontend，OpenAI 或原生 HTTP | Frontend 适配镜像。配置 `RTP_HOT_HOOK_CONFIG=/opt/rtp-frontend-capture/http.json` 和 `MOCK_CAPTURE_KEY`                |
| Python Frontend，DashScope gRPC     | Frontend 适配镜像。配置 `RTP_HOT_HOOK_CONFIG=/opt/rtp-frontend-capture/grpc.json` 和同一个 `MOCK_CAPTURE_KEY`          |
| 同一 Pod 承接两种入口               | 采用 `both.json`。HTTP 与 gRPC 进程使用不同的 worker 编号，只有 HTTP 首个 worker 监听采集控制端口                      |
| 协调 Frontend Pod                   | 设置 `MOCK_CAPTURE_EXPORT_VIP`。有远端 ViT 时设置 `MOCK_CAPTURE_VIT_EXPORT_VIP`。可用 `MOCK_CAPTURE_JOBS` 指定状态目录 |
| Mock ViT                            | 带采集代码的 Mock Engine 镜像。同一个 `MOCK_CAPTURE_KEY`。可用 `MOCK_CAPTURE_VIT_DIR` 将归档指向有余量的挂载目录       |
| Master、Prefill、Decode             | 沿用当前部署配置。核对真实路由和引擎缓存键口径。Decode 为 Mock 时实际输出保持缺省                                      |

`MOCK_CAPTURE_KEY` 是 32 字节随机密钥的十六进制表示，按部署密钥注入。设置有效配置路径后，Hook 在进程启动期间安装。使用 `legacy` 适配镜像时还需配置 `RTP_HOT_HOOK=1` 和 `RTP_HOT_HOOK_FILE=/opt/rtp-frontend-capture/hook.py`。采样任务通过内网接口装载窗口。可信真实 Decode 的采集可在 Frontend 设置 `MOCK_CAPTURE_REAL_OUTPUT=1`。

Frontend VIP 要覆盖全部目标 Pod。协调器从成员表取 IP，直接连接每个 Pod 的 **20557** 私网采集端口。ViT VIP 应登记每个 Pod 的实际 HTTP 控制端口，供协调器调用 `vit/capture/arm` 和 `vit/capture/read`。端口访问范围限于协调器网络。

如果挂载目录和默认目录不同，用适配镜像内的 `configure.py http|grpc|both <新配置路径> --directory <挂载目录>` 生成配置，在启动参数引用它。默认 Frontend 分片目录是 `/tmp/rtp-frontend-capture`。开始长批次前先做短批次，并按入口核对报告中的 `requests_openai_http`、`requests_native_http`、`requests_dashscope_grpc`。
