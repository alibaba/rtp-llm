# Bailian gRPC 使用说明

Bailian gRPC 在进程内提供 **predict_v2 协议**（`predict_v2.proto`）的 **`GRPCInferenceService` / `ModelStreamInfer`**（服务端流式）。HTTP Frontend（Uvicorn）与 Bailian gRPC **并行监听不同端口**；传输为 **明文 gRPC**（`insecure`），部署到公网时需自行做网络隔离或反向代理/TLS。

## 协议与访问方式

| 项目 | 说明 |
|------|------|
| Proto | `rtp_llm/bailian/proto/predict_v2.proto`（及 `model_config.proto`） |
| 服务名 | `GRPCInferenceService` |
| RPC | `ModelStreamInfer`（客户端发送一个或多个 `ModelInferRequest`，服务端流式返回 `ModelStreamInferResponse`） |
| 地址 | `0.0.0.0:<bailian_grpc_server_port>`（与下面端口计算一致） |

任意支持同一 `.proto` 的 gRPC 客户端均可调用；仓库内自带 Python 客户端（见下文）。

## 监听端口（随 Frontend 启动）

与 **Frontend 同进程**启动时，Bailian gRPC 端口由 `ServerConfig.bailian_grpc_server_port` 决定：

- **公式**：`start_port + rank_id * worker_info_port_num + 8`
- 默认 `start_port = 8088`，`worker_info_port_num` 至少为 `9` 时，**rank 0** 一般为 **`8088 + 8 = 8096`**（若你改了 `start_port` / `rank_id` / `worker_info_port_num`，按公式重算）。

### 升级注意：`worker_info_port_num` 默认值 8 → 9（破坏性）

为在每台 worker 的端口块内为 Bailian gRPC 预留 **base + 8** 且不与其他 rank 重叠，**`--worker_info_port_num` / `WORKER_INFO_PORT_NUM` 的默认值由 8 改为 9**。此前**未显式配置**该参数且使用 **多 rank / 分布式** 的部署，`rank ≥ 1` 的 **base 端口会整体偏移**，需同步改服务发现、防火墙或运维文档；若必须维持旧步进，可显式指定 `--worker_info_port_num 8`（须自行评估是否与新端口占用冲突）。

详见：[Breaking changes / `worker_info_port_num`](../release/breaking-changes.md)（含英文 Summary，便于写 release notes）。

启动 Frontend 后，日志中会出现类似 `Started Bailian gRPC server on port <port>`；也可用 `ServerConfig.to_string()` / 配置打印中的 `bailian_grpc_server_port` 确认。

## 启动方式

### 1. 随 RTP-LLM Frontend 启动（推荐）

正常启动带 Frontend 的 RTP-LLM 服务即可：在 **FastAPI/Uvicorn 启动阶段**会自动拉起 Bailian gRPC（后台线程 + 独立 `grpc.Server`）。

- **真实推理**：当 Frontend 已挂载 `FrontendWorker` 时，Bailian 请求会走 `backend_rpc_server_visitor.enqueue`，与主链路一致。
- **Fake（占位）**：若无 `FrontendWorker` / 无 backend visitor，服务端使用内置 mock（例如基于 `input_ids` 的简化输出），便于联调协议。

若启动失败，日志会提示检查 **`grpcio-tools`** 与 Python 桩是否已生成（见文末「开发：生成 Python proto」）。

### 2. 独立进程（仅 Fake 模式）

不拉起完整 Frontend、只做协议或客户端联调时，可单独起 Bailian gRPC 服务（**始终为 fake**，不接真实引擎）：

```bash
# 仓库根目录，PYTHONPATH 含 rtp_llm
python -m rtp_llm.bailian.bailian_grpc_server --port 8000
```

可选：与主服务相同形状的 JSON，覆盖通道选项与线程池（见下节）：

```bash
python -m rtp_llm.bailian.bailian_grpc_server --port 8000 \
  --bailian_grpc_config_json '{"client_config":{},"server_config":{},"max_server_workers":4}'
```

若使用 Bazel 打出的 **`rtp_llm_bailian_grpc` wheel**，入口点为：`rtp-llm-bailian-grpc`（等价于上述模块的 `main`）。

## 配置：`--bailian_grpc_config_json` / `BAILIAN_GRPC_CONFIG_JSON`

与 **Model RPC（C++）** 的 `--grpc_config_json` **相互独立**。Bailian 使用：

- **命令行**：`--bailian_grpc_config_json`
- **环境变量**：`BAILIAN_GRPC_CONFIG_JSON`

JSON 结构（逻辑上）包含：

- **`client_config`**：键为 gRPC channel option 名，值为整数（Python 客户端建连时使用）。
- **`server_config`**：服务端 `grpc.server(..., options=...)` 的选项。
- **`max_server_workers`**：服务端 `ThreadPoolExecutor` 大小，须 **> 0**（默认一般为 **4**）。

主程序解析后写入 `BailianGrpcConfig`（C++/pybind 与 Python 侧一致）。**不要**把 Model RPC 专用的 `max_server_pollers` 当成 Bailian 服务端线程数；二者用途不同。

## 使用自带 Python 客户端访问

客户端模块：`rtp_llm.bailian.bailian_grpc_client`。需本地 tokenizer 与 checkpoint 路径与 Frontend 一致（`TokenizerFactory`）。

```bash
python -m rtp_llm.bailian.bailian_grpc_client \
  --grpc_addr 127.0.0.1:<bailian_grpc_server_port> \
  --ckpt_path /path/to/checkpoint \
  --model_type qwen2 \
  --prompt "Hello"
```

常用参数：`--tokenizer_path`（默认与 `ckpt_path` 相同）、`--request_id`、`--model_name`，以及 `--max_new_tokens`、`--top_k`、`--top_p`、`--temperature` 等采样参数。若服务端选项与默认不一致，可传 `--bailian_grpc_config_json` 以匹配 channel 的 `client_config`。

仓库内还提供 Bash 封装（**必须用 bash**）：

```bash
cd rtp_llm/bailian
export GRPC_ADDR=127.0.0.1:<bailian_grpc_server_port>
bash grpc_client_run.sh
```

脚本会通过环境变量设置 `PYTHON`、`CKPT_PATH`、`MODEL_TYPE`、`PROMPT` 等；支持压测循环 `GRPC_CLIENT_LOOPS`、`GRPC_CLIENT_DELAY_SEC`。详见脚本内注释。

## 开发：生成 Python proto

修改 `.proto` 后，在**仓库根**执行：

```bash
python -m rtp_llm.bailian.generate_proto_py
```

依赖 **`grpcio-tools`**。生成文件位于 `rtp_llm/bailian/proto/`。

## 相关代码路径（便于深入）

- 服务实现：`rtp_llm/bailian/bailian_grpc_server.py`
- 请求解析 / 张量约定：`rtp_llm/bailian/bailian_grpc_request.py`
- 客户端：`rtp_llm/bailian/bailian_grpc_client.py`
- Frontend 挂载：`rtp_llm/frontend/frontend_app.py`（`start_bailian_grpc_server_in_thread`）
- 参数定义：`rtp_llm/server/server_args/grpc_group_args.py`（`init_bailian_grpc_group_args`）

## 单测

```bash
bazel test //rtp_llm/bailian:bailian_grpc_request_test
bazel test //rtp_llm/bailian:bailian_grpc_response_server_test
```

`bailian_grpc_response_server_test` 覆盖 `build_stream_response_from_generate_outputs`、`iter_real_model_stream_infer`（mock `run_enqueue_sync`）以及 `BailianGrpcInferenceServicer.ModelStreamInfer`（fake / real 分支与缺 `input_ids` 错误路径）。
