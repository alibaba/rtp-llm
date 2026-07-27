# DashSc gRPC 使用说明

DashSc gRPC 在进程内提供 **predict_v2 协议**（`predict_v2.proto`）的 **`GRPCInferenceService` / `ModelStreamInfer`**（服务端流式）。HTTP Frontend（Uvicorn）与 DashSc gRPC **并行监听不同端口**；传输为 **明文 gRPC**（`insecure`），部署到公网时需自行做网络隔离或反向代理/TLS。

## 协议与访问方式

| 项目 | 说明 |
|------|------|
| Proto | `rtp_llm/dash_sc/proto/predict_v2.proto`（及 `model_config.proto`） |
| 服务名 | `GRPCInferenceService` |
| RPC | `ModelStreamInfer`（客户端发送一个或多个 `ModelInferRequest`，服务端流式返回 `ModelStreamInferResponse`） |
| 地址 | `0.0.0.0:<dash_sc_grpc_server_port>`（与下面端口计算一致） |

任意支持同一 `.proto` 的 gRPC 客户端均可调用；仓库内自带 Python 客户端（见下文）。

## 监听端口（随 Frontend 启动）

与 **Frontend 同进程**启动时，DashSc gRPC 端口由 `ServerConfig.dash_sc_grpc_server_port` 决定：

- **公式**：`start_port + rank_id * worker_info_port_num + 8`
- 默认 `start_port = 8088`，`worker_info_port_num` 至少为 `9` 时，**rank 0** 一般为 **`8088 + 8 = 8096`**（若你改了 `start_port` / `rank_id` / `worker_info_port_num`，按公式重算）。

### 升级注意：`worker_info_port_num` 默认值 8 → 9（破坏性）

为在每台 worker 的端口块内为 DashSc gRPC 预留 **base + 8** 且不与其他 rank 重叠，**`--worker_info_port_num` / `WORKER_INFO_PORT_NUM` 的默认值由 8 改为 9**。启用 DashSc gRPC 的非 VIT 服务要求 `worker_info_port_num >= 9`，显式配置为 `8` 会在启动校验阶段失败。此前使用 **多 rank / 分布式** 且依赖旧步进的部署，`rank ≥ 1` 的 **base 端口会整体偏移**，需同步改服务发现、防火墙或运维文档。

详见：[Breaking changes / `worker_info_port_num`](../release/breaking-changes.md)（含英文 Summary，便于写 release notes）。

启动 Frontend 后，日志中会出现类似 `Started DashSc gRPC server on port <port>`；也可用 `ServerConfig.to_string()` / 配置打印中的 `dash_sc_grpc_server_port` 确认。

## 启动方式

### 1. 随 RTP-LLM Frontend 启动（推荐）

正常启动带 Frontend 的 RTP-LLM 服务即可：在 **FastAPI/Uvicorn 启动阶段**会自动拉起 DashSc gRPC（后台线程 + 独立 `grpc.Server`）。

- **真实推理**：DashSc 请求会走 `backend_rpc_server_visitor.enqueue`，与主链路一致。

若启动失败，日志会提示检查 **`grpcio-tools`** 与 Python 桩是否已生成（见文末「开发：生成 Python proto」）。

### 2. 独立反向代理进程

不拉起完整 Frontend、只做 gRPC 反向代理或 canary 联调时，可单独起 DashSc gRPC proxy。proxy 会按 `SERVICE_ROUTE`（或兼容的 `DASH_SC_GRPC_FORWARD_ADDR`）把请求转发到下游 Frontend 的 DashSc gRPC 端口。

```bash
export SERVICE_ROUTE='{"type":"ip_port_list","address":"127.0.0.1:8088"}'
python -m rtp_llm.dash_sc.server --port 8000
```

可选：与主服务相同形状的 JSON，覆盖通道选项与线程池（见下节）：

```bash
python -m rtp_llm.dash_sc.server --port 8000 \
  --dash_sc_grpc_config_json '{"client_config":{},"server_config":{},"max_server_workers":4}'
```

## 配置：`--dash_sc_grpc_config_json` / `DASH_SC_GRPC_CONFIG_JSON`

与 **Model RPC（C++）** 的 `--grpc_config_json` **相互独立**。DashSc 使用：

- **命令行**：`--dash_sc_grpc_config_json`
- **环境变量**：`DASH_SC_GRPC_CONFIG_JSON`

JSON 结构（逻辑上）包含：

- **`client_config`**：键为 gRPC channel option 名，值为整数（Python 客户端建连时使用）。
- **`server_config`**：服务端 `grpc.server(..., options=...)` 的选项。
- **`max_server_workers`**：服务端 `ThreadPoolExecutor` 大小，须 **> 0**（默认一般为 **4**）。

主程序解析后写入 `DashScGrpcConfig`（C++/pybind 与 Python 侧一致）。**不要**把 Model RPC 专用的 `max_server_pollers` 当成 DashSc 服务端线程数；二者用途不同。

## 使用自带 Python 客户端访问

客户端模块：`rtp_llm.dash_sc.client`。需本地 tokenizer 与 checkpoint 路径与 Frontend 一致（`TokenizerFactory`）。

```bash
python -m rtp_llm.dash_sc.client \
  --grpc_addr 127.0.0.1:<dash_sc_grpc_server_port> \
  --ckpt_path /path/to/checkpoint \
  --model_type qwen2 \
  --prompt "Hello"
```

常用参数：`--tokenizer_path`（默认与 `ckpt_path` 相同）、`--request_id`、`--model_name`，以及 `--max_new_tokens`、`--top_k`、`--top_p`、`--temperature` 等采样参数。若服务端选项与默认不一致，可传 `--dash_sc_grpc_config_json` 以匹配 channel 的 `client_config`。

DeepSeek-V4 的 dash-sc 请求是预 tokenized wire。Python 客户端只做 raw-token 调试：`tokenizer.encode(prompt)` 后发送 `input_ids`。真实 chat prompt 渲染、tool_choice 语义和 reasoning 参数归一化应由 OpenAI / DashScope 前端链路完成，dash-sc gRPC 层只承接已编码的 `input_ids` 和 generation 参数。

### DeepSeek-V4 tool-call guided decoding

当前版本的 DashSc gRPC 尚未接入引擎侧 grammar backend。上游传入结构化输出/grammar 约束时会在入口 fail-fast 返回参数错误，避免请求被静默接受但输出不受约束：

- 直接参数：`request.parameters["tool_call_structural_tag"]`
- 兼容别名：`request.parameters["structural_tag"]`
- DashScope header 兼容：`ds_header_attributes.parameters.tool_call_structural_tag` / `structural_tag`

同样会拒绝 `response_format`、`guided_json` 和 `json_format`。若后续引擎侧 grammar backend 落地，再恢复这些参数的透传和编译校验。

普通生成请求仍可用客户端调试；不要传 `--response_format`、`--json_format` 或 `--tool_call_structural_tag`：

```bash
python -m rtp_llm.dash_sc.client \
  --grpc_addr 127.0.0.1:<dash_sc_grpc_server_port> \
  --ckpt_path /path/to/model \
  --model_type deepseek_v4 \
  --prompt "<already-rendered-prompt-or-raw-debug-text>" \
  --max_new_tokens 64 \
  --temperature 0 \
  --top_k 1 \
  --enable_thinking false
```

仓库内还提供 Bash 封装（**必须用 bash**）：

```bash
cd rtp_llm/dash_sc
export GRPC_ADDR=127.0.0.1:<dash_sc_grpc_server_port>
export CKPT_PATH=/path/to/model
bash grpc_client_run.sh
```

脚本会通过环境变量设置 `PYTHON`、`CKPT_PATH`、`MODEL_TYPE`、`PROMPT` 等，其中 `CKPT_PATH` 必须显式设置为本地模型路径。Python client 是低层 gRPC 调试工具，只执行 `tokenizer.encode(prompt)` 后发送 `input_ids`；OpenAI / DashScope chat 渲染应在上游完成。支持压测循环 `GRPC_CLIENT_LOOPS`、`GRPC_CLIENT_DELAY_SEC`。详见脚本内注释。

## 开发：生成 Python proto

修改 `.proto` 后，在**仓库根**执行：

```bash
python -m rtp_llm.dash_sc.generate_proto_py
```

依赖 **`grpcio-tools`**。生成文件位于 `rtp_llm/dash_sc/proto/`。

## 相关代码路径（便于深入）

- 服务生命周期：`rtp_llm/dash_sc/server.py`、`rtp_llm/dash_sc/app.py`
- 推理 / 代理 servicer：`rtp_llm/dash_sc/inference/servicer.py`、`rtp_llm/dash_sc/proxy/servicer.py`
- 请求解析 / 张量约定 / 响应构建：`rtp_llm/dash_sc/codec.py`
- 客户端：`rtp_llm/dash_sc/client.py`
- 参数定义：`rtp_llm/server/server_args/grpc_group_args.py`（`init_dash_sc_grpc_group_args`）

## 单测

```bash
bazel test //rtp_llm/dash_sc/test:codec_test
bazel test //rtp_llm/dash_sc/test:inference_servicer_test
bazel test //rtp_llm/dash_sc/test:proxy_servicer_test
bazel test //rtp_llm/dash_sc/test:access_log_test
```

`codec_test` 覆盖请求解析、`SamplingParams` / `DashScRequestControls` 以及 `build_stream_response_from_generate_outputs`；
`inference_servicer_test` 覆盖 `iter_real_model_stream_infer`（mock `run_enqueue_sync`）、`DashScInferenceServicer.ModelStreamInfer` 与缺 `input_ids` 错误路径；
`proxy_servicer_test` 覆盖 gRPC proxy 转发、下游异常和流关闭路径。
