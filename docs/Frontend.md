## Frontend

### Overview
RTP_LLM currently comprises three core components: Frontend, Backend, and Master.

Frontend Workflow:
* Accepts incoming requests
* Converts inputs to token IDs (includes tokenizer decoding and OpenAI request rendering)
* Queries the Master to obtain Backend IP
* Submits requests to Backend and awaits responses
* Processes responses (includes tokenizer encoding and function call rendering)
### Role Initialization

``` python
class RoleType(Enum):
    PDFUSION = 0  # Monolithic mode
    PREFILL = 1
    DECODE = 2
    VIT = 3
    FRONTEND = 4
```
The active role is determined by the ROLE_TYPE environment variable (default: PDFUSION). Other roles only launch the corresponding component.

In frontend only deployments, engine initialization is skipped for rapid tokenizer/renderer debugging.

Backend servers still host Frontend apps (for health checks/debugging).

*Italicized* APIs below are only usable when locally paired with a Backend server.

---

### Public APIs
#### Health Check Endpoints
Verifies Backend status (returns ok/error). Call same endpoints in Backend.
``` python
@app.get("/health")
@app.post("/health")
@app.get("/GraphService/cm2_status")
@app.post("/GraphService/cm2_status")
@app.get("/SearchService/cm2_status")
@app.post("/SearchService/cm2_status")
@app.get("/status")
@app.post("/status")
@app.post("/health_check")

@app.get("/")
```
#### *Debug Endpoints*
Proxied to same endpoints in Backend.
```python
@app.get("/cache_status")
@app.post("/cache_status")
@app.get("/rtp_llm/cache_status")
@app.post("/rtp_llm/cache_status")

# input
class WorkerStatusRequest(BaseModel):
    lastest_cache_version: Optional[int] = -1

# output
class CacheStatus(BaseModel):
    available_kv_cache: int = -1
    total_kv_cache: int  = -1
    block_size: int = -1
    version: int = -1
    cached_keys: Optional[List[int]] = None
```

``` python
@app.get("/worker_status")
@app.post("/worker_status")
@app.get("/rtp_llm/worker_status")
@app.post("/rtp_llm/worker_status")

# input
class WorkerStatusRequest(BaseModel):
    lastest_cache_version: Optional[int] = -1
    latest_finised_version: Optional[int] = -1

# output
class WorkStatus(BaseModel):
    role: str  # prefill, decode, vit
    server_port: int
    http_proto_port: int
    grpc_proto_port: int
    available_concurrency: int

    running_task_info: List[TaskInfo]
    finished_task_list: List[TaskInfo]

    step_latency_ms: float
    iterate_count: int

    dp_size: int
    tp_size: int
    alive: bool
    version: int
    cache_status: Optional[CacheStatus] = None
    profile_meta: Optional[ProfileMeta] = None
```

#### *Dynamic Update Endpoints*
Proxied to same endpoints in Backend.

``` python
@app.post("/update")
# example : {"peft_info": {"lora_info": {"lora_0": "/lora/llama-lora-test/""}}}

# input
class VersionInfo(BaseModel):
    models_info: Optional[Dict[str, str]] = None
    peft_info: Optional[Dict[str, Any]] = None
    sampler_info: Optional[Dict[str, Any]] = None

# output:
# error info when failed
```

``` python
@app.post("/set_log_level")
# request format: {"log_level": "DEBUG/INFO/TRACE/WARNING"}
```

``` python
@app.post("/update_eplb_config")
# request format: {"mode": "NONE", "update_time": 5000}

# input:
class EplbMode(Enum):
    NONE
    STATS  # stats, only
    EPLB   # load balance, only
    ALL    # stats + load balance

class EplbConfig:
  mode: EplbMode
  update_time: int
```

#### *Embedding APIs*
Proxied to same endpoints in Backend.

```
python
@app.post("/v1/embeddings")
@app.post("/v1/embeddings/dense")
@app.post("/v1/embeddings/sparse")
@app.post("/v1/embeddings/colbert")
@app.post("/v1/embeddings/similarity")
@app.post("/v1/classifier")
@app.post("/v1/reranker")
```

#### Inference APIs
``` python
@app.post("/")
# input
# prompt: str
# urls: optional[List[str]]
# generate_config: GenerateConfig

# output
# inference result
```

``` python
@app.post("/chat/completions")
@app.post("/v1/chat/completions")

# input
class ChatCompletionRequest(BaseModel):
  model: Optional[str] = None
  messages: List[ChatMessage]
  functions: Optional[List[GPTFunctionDefinition]] = None
  tools: Optional[List[GPTToolDefinition]] = None
  temperature: Optional[float] = 0.7
  top_p: Optional[float] = 1.0
  max_tokens: Optional[int] = None
  stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
  stream: Optional[bool] = False
  user: Optional[str] = None
  seed: Optional[int] = None
  n: Optional[int] = None
  logprobs: Optional[bool] = None
  top_logprobs: Optional[int] = None

  # ---- These functions are not implemented yet.
  # presence_penalty: Optional[float] = 0.0
  # frequency_penalty: Optional[float] = 0.0
  # logit_bias: Optional[Dict[str, float]] = None

  # ---- These params are hacked for our framework, not standard.
  extra_configs: Optional[GenerateConfig] = None
  private_request: bool = False
  trace_id: Optional[str] = None
  chat_id: Optional[str] = None
  template_key: Optional[str] = None
  user_template: Optional[str] = None
  debug_info: Optional[bool] = False
  aux_info: Optional[bool] = False
  extend_fields: Optional[Dict[str, Any]] = (
    None  # This field is not effective, only for logging.
  )
  master_info: Optional[Dict[str, Any]] = None
  chat_template_kwargs: Optional[Dict[str, Any]] = None

# output
# inference response
```
#### Prompt Processing APIs
``` python
@app.post("/chat/render")
@app.post("/v1/chat/render")

# input
class ChatCompletionRequest:
    ...

# output
class DebugInfo(BaseModel):
    input_prompt: str
    input_ids: List[int]
    input_urls: List[str]
    tokenizer_info: str
    max_seq_len: int
    eos_token_id: Optional[int]
    stop_word_ids_list: List[List[int]]
    stop_words_list: List[str]
    renderer_info: RendererInfo
    generate_config: GenerateConfig
```

``` python
@app.post("/tokenizer/encode")
# input
# prompt: str
# return_offsets_mapping: bool

# output
class TokenizerEncodeResponse(BaseModel):
    token_ids: List[int] = []
    offset_mapping: Optional[List[Any]] = None
    tokens: List[str] = []
    error: str = ""

@app.post("/tokenize")
# input
# raw or openai request

# output
# token ids
```

---

### Internal Communication

Frontend → Master: HTTP call to obtain Backend IP.

Frontend → Backend: gRPC call for inference (see model_rpc_service.proto).

#### Master APIs
``` python
class RoleType(Enum):
    PDFUSION = 0
    PREFILL = 1
    DECODE = 2
    VIT = 3

class ServerStatus(BaseModel):
    role: RoleType
    server_ip: str
    http_port: int
    grpc_port: int
    debug_info: Optional[DebugInfo]

class ScheduleMeta(BaseModel):
    server_status: List[ServerStatus]
    cache_local: int          # 0: LOCAL, 1: REMOTE
    inter_request_id: int

@app.post("/rtp_llm/master")
# "real_master_host": "{master_ip}:{port}"

@app.post("/rtp_llm/schedule")
# input
# model: str
# block_cache_keys: list[int]
# seq_len: int
# debug: bool
# generate_timeout: int
# request_priority: int

# output
# ScheduleMeta
```

#### Backend grpc APIs

``` python
GenerateStreamCall
# input
# GenerateInputPB

# output
# GenerateOutputsPB
```

---

### Debugging Procedures

#### Frontend-Only Deployment
``` bash
MODEL_SERVICE_CONFIG='{"service_id":"test","master_endpoint":{"type":"VipServer","address":"127.0.0.1:16000","protocol":"http","path":"/"},"use_local":true}' \
ROLE_TYPE=FRONTEND \
START_PORT=12345 \
TOKENIZER_PATH=/mnt/nas1/hf/Qwen2-0.5B-Instruct/ \
MODEL_TYPE=qwen_2 \
FRONTEND_SERVER_COUNT=1 \
/opt/conda310/bin/python -m rtp_llm.start_server
```

No ckpt_path required. Test tokenizers/renderers via prompt processing APIs.

#### Post-Processing Debugging
Frontend defaults use localhost:start_port+1 for gRPC call.

Mock a Master & Backend for function call testing:
``` python
# master.py
import asyncio
from enum import Enum
import uvicorn
import os

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FastAPI 应用
app = FastAPI()

# 添加 CORS 支持（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebugInfo(BaseModel):
    running_batch_size: int
    queue_size: int
    waiting_time_ms: int
    available_kv_cache_len: int
    estimate_ttft_ms: int
    estimate_ttot_ms: int
    hit_cache_len: int


class RoleType(Enum):
    PDFUSION = 0
    PREFILL = 1
    DECODE = 2
    VIT = 3

class ServerStatus(BaseModel):
    role: RoleType
    server_ip: str
    http_port: int
    grpc_port: int
    debug_info: Optional[DebugInfo]


class ScheduleMeta(BaseModel):
    server_status: List[ServerStatus]
    cache_local: int          # 0: LOCAL, 1: REMOTE
    inter_request_id: int

class ScheduleRequest(BaseModel):
    block_ids: list[int]
    seq_len: int
    debug: bool

@app.post("/rtp_llm/master")
async def http_master(request: BaseModel):
    return {"real_master_host": "127.0.0.1:16000"}

# FastAPI HTTP 接口
@app.post("/rtp_llm/schedule")
async def http_schedule(request: BaseModel):
    server_status_list: List[ServerStatus] = []
    print("http call schedule")
    # port_list = [26000]  # 模拟服务器 IP 列表
    # for port in port_list:
    server_status = ServerStatus(
        role=RoleType.PDFUSION,
        server_ip="127.0.0.1",
        http_port=26000,
        grpc_port=26000 + 1,
        debug_info=None,
    )
    server_status_list.append(server_status)

    # server_status = ServerStatus(
    #     role=RoleType.DECODE,
    #     server_ip="127.0.0.1",
    #     http_port=27000,
    #     grpc_port=27000 + 1,
    #     debug_info=None,
    # )
    # server_status_list.append(server_status)
    return ScheduleMeta(
        server_status=server_status_list, cache_local=0, inter_request_id=1
    )

async def main():
    mock_master_port = int(os.environ.get("MOCK_MASTER_PORT", 16000))
    config = uvicorn.Config(app, host="0.0.0.0", port=mock_master_port, loop="asyncio")
    server = uvicorn.Server(config)
    await asyncio.gather(
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
```

Also, you can start server with ROLE_TYPE=PDFUSION to start backend server engine.

In this way, debugging the tokenizer and openai renderer related code only requires restarting frontend (lightweight).