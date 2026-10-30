# GLM5/GLM53 hidden-state capture：DSpark / TorchSpec 接入

## 来源与范围

- 移植来源：main #1312 `e73826e27`、#1526 `bcd02fdb0` 的相关能力；base `3adc4f1f`。这是适配性移植，不包含原 PR 的无关量化改动，也不引入 main 的 scheduler/batch RPC 前置重构。
- 当前主树分支：`feat/glm5_cu13_rebase_hidden_state_for_glm`；内源 `internal_source` 分支：`feat/glm5_cu13_rebase`（均已读取分支名确认）。
- 目标：GLM53 target prefill 采集 → TorchSpec/Mooncake 传输 → draft 训练。保持 main 的数据/完成语义契约，但适配本 base 的现有模型、调度和 RPC，不整体替换为 main 架构。
- 模型精度历史见 [GLM5 hidden 对齐日志](glm5_hidden_align_log.md)，不在此复制历史模型路径、层数、实验结果，也不将其视为本移植的验证结果。

## 配置与请求

真实入口为服务参数 `--json_model_override_args`，对应环境变量 `JSON_MODEL_OVERRIDE_ARGS`；内容由 `ModelConfig.apply_override_args()` 校验。CLI 与环境变量不要配置成互相矛盾的值。

| 字段 | 契约 |
| --- | --- |
| `hidden_state_capture_layer_ids` | 整数列表，默认 `[]` 关闭；0-based decoder 层号，保持配置顺序，不排序；不允许重复、负数、越界或布尔值 |
| `hidden_state_capture_dtype` | `"bf16"`（默认）或 `"fp8_e4m3"`；独立于模型权重的量化类型 |

下例仅展示配置形状，**不是 GLM53 的真实层选择，也不是可直接执行的配置**；需以实际 target/draft 配置替换全部尖括号占位：

```text
JSON_MODEL_OVERRIDE_ARGS='{"hidden_state_capture_layer_ids":[<已确认的0-based层号，按draft输入顺序>],"hidden_state_capture_dtype":"bf16"}'
```

请求使用 `GenerateConfig(max_new_tokens=0)`；不设置 `return_hidden_states` / `return_all_hidden_states`。两者在 prefill-only 下会被拒绝，不是采集开关。此路径跳过 LM head/采样，不产生新 token；`return_logits`、非零 `calculate_loss`、beam/多返回序列等也受 prefill-only 校验限制。训练 loss 在下游计算，不用推理端 `calculate_loss` 代替。

Python RPC client 会把 prefill-only 写入 `GenerateConfigPB.prefill_only`，并检查响应没有生成 token；手写客户端应同时传 `max_new_tokens=0` 和 `prefill_only=true`，不要依赖旧服务对 proto3 默认零值的理解。本 base 的 `QueryConverter` 也接受未标志的零预算；`prefill_only=true` 搭配非零预算会被拒绝。

本移植不依赖 main 的 capture batch RPC，普通请求仍走原单请求 `GenerateStreamCall`，并发由现有调度聚批；已由 master 入队的路径保留 `FetchResponse`。仓库另有 `EnqueueBatch` / `EnqueueGroup`，不能将“未移植 main batch RPC”理解为完全不存在任何 batch 接口。

## 数据、标识与完成语义

令 `L` 为选中层数、`H` 为实际 target hidden size、`T` 为恢复原始顺序后的有效 token 数：

- GLM `generic_moe.py` 每个 decoder 层采集 **post-residual**：`hidden_states + residual`。这不是层输入，也不是已做最终 norm 的 hidden；不要直接套用含 embedding 项的 HF hidden-state tuple 下标。
- 最终输出单独为 `norm(hidden_states, residual)[0]`。内部 packed 布局 `[T, (L+1)*H]`：前 `L` 块按配置层号顺序排列，最后一块为 normalized final。
- 发布前按每个请求的原始长度切片，CP 恢复原 token 顺序，去掉分布式 padding；`input_ids` 为同序的连续 int64 张量。

| 存储字段（单请求长度 `t`） | BF16 capture | FP8 capture |
| --- | --- | --- |
| `hidden_states` | `[t, L*H]` BF16 | `[t, L*H]` float8_e4m3fn |
| `last_hidden_states` | `[t, H]` BF16 | `[t, H]` BF16，不能量化替代 |
| `hidden_states_scale` | 不传 | `[t, L]` FP32，与 auxiliary 同设备 |
| `input_ids` | `[t]` int64 | `[t]` int64 |

`MooncakeConfig.hidden_dim` 必须等于 `H`，`num_aux_layers` 必须等于 `L`。FP8 通过 `quantize_aux_hidden_states(auxiliary, L)` 生成张量和 scale；消费者必须使用匹配的反量化契约，不能只把 FP8 cast 成 BF16。

存储 key 由 **同一份** `MooncakeConfig.make_store_key(request_id)` 产生，包含 namespace 与 ID 的约定；不要在消费者手写 key 格式。request ID 必须非负、同批唯一，跨请求/重试/运行还需避免 namespace 内碰撞。RTP batch 诊断 ID 含 namespace、递增序号及首尾请求 ID，但不是逐请求 store key。仅 TP rank 0 拥有 publisher；不要让每个 rank 重复发布。

`put_batch(...)` 返回逐请求 accepted metadata 列表，仅代表 **admission**，不是异步传输完成。当前 publisher 随后调用 `flush()` 并读取 `take_async_errors()`，之后才结束本次发布；关闭时也 flush/close。消费者应遵循匹配 TorchSpec 的可读/完成契约，不把 accepted 列表当作训练样本已完整可读的证明。

错误也不能静默当作成功采样：硬布局/接口契约错误会禁用后续 capture；请求元数据或操作失败有独立处理。历史异步错误保留原归属并记录日志/指标，不应错误归到新请求。联测检查 existing capture status/failure 指标，以及 `hidden_state_capture.put_batch_admit`、`flush`、`take_async_errors` profiling 范围，区分 admission 耗时和完成耗时。

## GLM 路径边界

- 保留 GLM CMP、CP、MTP hidden handoff/indexer seed 等 base 路径，普通生成不因配置了 capture 就改变输出布局。
- capture forward 强制 eager，绕开 CUDA Graph replay；不是关闭所有普通生成的 CUDA Graph。
- **capture 不支持 CP + layer micro-batch 的组合**，初始化会拒绝。CP 联测使用真实参数 `--enable_layer_micro_batch 0` 或 `ENABLE_LAYER_MICRO_BATCH=0`；不是宣称 CP 本身不受支持。
- capture plan 只装配到 TARGET 的 PREFILL/PDFUSION 角色；DRAFT、DECODE、warmup 不启用。fake stream 不发布。MTP 普通推理仍需回归，不能用训练采集成功替代 MTP 验证。

### GLM prefill CP 专项

当前 ZigZag CP 使用 TP group：CP size/rank 对应 `TP_SIZE`/TP rank。`CP_ROTATE_METHOD=ALL_GATHER` 等启用该路径；`PREFILL_CP_SIZE` 不独立控制这里的切分宽度。线上参数需与实际部署一致，不能仅开启一个名为 prefill 的选项就假定进入 ZigZag。

每个请求独立 padding 后分为前后两段；attention 内部可以按有效 Q 重排，但返回模型层前必须 scatter 回 rank-local padded 行。各层 capture 保留该行布局，最后统一 gather、恢复全局请求顺序、去 padding，使用原始请求长度和 token IDs 发布。TP0 是唯一 publisher owner，其他 rank 仍必须参加恢复所需的 collective。

本次专项修复与回归：

- CP 开启、TP=1 时，恢复直接使用本地 tensor 并去 padding，不读取 no-op all-gather 的未初始化接收缓冲区。
- CP capture 的 packed layout 与可选 MTP getter 状态使用同一次 TP 共识。getter 异常、非对称 presence、错误宽度/dtype/device 在恢复 gather 前统一拒绝。GLM 没有该 pre-norm getter；它继续使用恢复后的 normalized hidden，不能套用 DSV4 的 expanded pre-norm handoff。
- CP 拒绝路径返回全局形状的零 fallback，避免局部行数偶然等于全局 token 数时泄漏局部顺序或 padding。
- 单 GPU 回归覆盖真实 C++ split/restore 的 CP2/CP4 异长批、NaN padding、全部 padding 的 rank，以及逐请求 capture/ID 对齐；通信边界使用替身。
- GLM53 kernel 回归用真实 CP metadata 覆盖 `[256,512]` 和 `[257,513]` 的 Q selection 与延迟 scatter 正反分支。
- 独立双 GPU 用例使用真实 NCCL、C++ split/restore 和 publisher，覆盖正常发布及非 owner 故障后的通信推进；模型层输出为合成数据，Mooncake store 为替身，不需要完整 GLM 权重，也不替代真实模型精度验证。

**长上下文容量限制尚未优化：** capture 仍在每个 CP rank 对完整 packed tensor 做 all-gather，再用 `index_select` 生成恢复结果。两份全局 buffer 的主要占用约为 `2 * T * (L+1) * H * element_size`，还不含 padding、本地 packed、权重、KV 和工作区。例：`T=2^20, H=6144, L=3, BF16, CP=8`，无 padding 时仅两份全局 buffer 加本地 packed 就约 **102 GiB/rank**。这是容量计算，不是实测；FP8 在发布阶段量化，不会降低此前 CP 恢复峰值。线上最大 prompt/batch 必须按这一额外开销验证，不能沿用普通 last-hidden-only 推理的显存上限。

## TorchSpec 运行前置条件与待确认输入

RTP publisher 在同一 Python 运行环境实际调用以下接口：

1. `torchspec.transfer.mooncake.MooncakeConfig.from_env()`；配置暴露 `store_key_namespace`、`make_store_key()`、`hidden_dim`、`num_aux_layers`。
2. `EagleMooncakeStore(config)`，支持 `setup(torch.device("cuda:<local_rank>"))`、`warmup_rdma()`、`put_batch(batch_id=..., request_ids=..., hidden_states=..., input_ids=..., last_hidden_states=..., hidden_states_scale=...)`（scale 仅 FP8）、`take_async_errors()`、`flush()`、`close()`。
3. FP8 额外要求 `torchspec.utils.fp8.quantize_aux_hidden_states()`；返回布局需满足上表。

还需匹配的 PyTorch/CUDA、RTP native bindings、TorchSpec 所选 Mooncake 客户端及可用 store/RDMA 服务。当前环境没有可读取的 TorchSpec checkout，不能据此确定发行版本或 `from_env()` 的环境变量名；**不支持猜版本安装，也不编造 `MOONCAKE_*` / `TORCHSPEC_*` 变量**。部署前从选定 TorchSpec commit 的配置/依赖元数据确认 endpoint、namespace、布局及资源参数，并记录双方 commit/环境。

启动真实 GLM53 draft 训练前必须提供/核对：

- target checkpoint、draft checkpoint/config、tokenizer 的实际位置和 revision；draft 所需 target 层号、0-based/其他编号转换、顺序、`L`/`H`，不能使用测试中的示例层号。
- tokenizer vocabulary / special token ID / chat template / BOS/EOS / padding 与 truncation 一致；逐请求 `input_ids` 和 hidden 行逐一对应。
- labels 的 next-token shift、loss mask（prompt/assistant/padding/截断边界）、样本边界与训练目标定义；publisher 不生成训练 labels 或 mask。
- reader 对 namespace/ID、异步完成、重试去重和 FP8 scale 的实现。先证明训练 forward/loss 输入对齐，再扩大采集规模。

## 另一台 CUDA 机器的验证顺序

以下 target 已按当前 BUILD 核对，但重写后的测试尚未运行。在 CUDA 13 / SM10x（10.0、10.3）机器、对应 checkout 根目录执行；使用 `bazelisk`，不要混入 `--config=ppu` 或 `--config=sm120`。分组从窄到宽运行，并由执行器设置有限总超时，超时后人工决定是否重跑。

```sh
bazelisk test --config=cuda13 --config=sm10x \
  --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0,10.3 \
  --test_output=errors --test_timeout=600 \
  //rtp_llm/config/test:model_config_override_test \
  //rtp_llm/server/server_args/test:server_args_test \
  //rtp_llm/test:generate_config_test \
  //rtp_llm/test:generate_config_validators_test \
  //rtp_llm/models_py/model_desc/test:hidden_state_capture_test \
  //rtp_llm/cpp/model_rpc/test:rpc_error_code_proto_test

bazelisk test --config=cuda13 --config=sm10x \
  --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0,10.3 \
  --test_output=errors --test_timeout=600 \
  //rtp_llm/cpp/models/test:pywrapped_model_cache_store_integration_test \
  //rtp_llm/cpp/model_rpc/test:model_rpc_client_test \
  //rtp_llm/cpp/model_rpc/test:query_converter_test \
  //rtp_llm/cpp/model_rpc/test:rpc_writer_cancellation_test \
  //rtp_llm/cpp/engine_base/stream/test:generate_stream_test \
  //rtp_llm/cpp/engine_base/schedulers/test:fifo_scheduler_test \
  //rtp_llm/cpp/engine_base/schedulers/test:gather_batch_scheduler_test \
  //rtp_llm/cpp/normal_engine/test:batch_stream_processor_test \
  //rtp_llm/cpp/normal_engine/test:engine_test \
  //rtp_llm/cpp/normal_engine/speculative/test:mtp_executor_test
```

CP 专项先执行单 GPU 的真实布局/kernel 回归，再执行独立双 GPU collective 回归：

```sh
bazelisk test --config=cuda13 --config=sm10x \
  --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0,10.3 \
  --test_output=errors --test_timeout=600 \
  //rtp_llm/cpp/models/context_parallel/test:context_parallel_py_wrapper_test \
  //rtp_llm/cpp/models/test:pywrapped_model_cache_store_integration_test \
  //rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:glm53_prefill_q_selection_test

# 需要两张可见 GPU；测试内部 spawn ranks，不要再套 torchrun。
bazelisk test --config=cuda13 --config=sm10x \
  --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0,10.3 \
  --test_output=errors --test_timeout=600 \
  //rtp_llm/cpp/models/test:pywrapped_model_cp_capture_test
```

双 GPU target 缺少两张 GPU 时会失败，不静默跳过；仅测试 BF16，不需要实际 Mooncake 服务或 FP8 quantizer。其执行平台属性为 `SM103_CU13`、`gpu_count=2`，远端调度需具备对应资源。

publisher integration target 标记 `manual`，这里显式指定以免被通配遗漏。测试调用真实 C++ publisher / PyWrappedModel、torch Tensor，以及 FP8 成功路径的真实 TorchSpec quantizer；目标环境必须提供匹配的 TorchSpec 包。Mooncake store、远端 TP 通信和故障注入仍使用边界替身，不替代真实 RDMA、多 rank 或完整模型验证。部分 BUILD 声明了 H20 execution property，使用远端执行平台时需核对其与 SM10x 验证机器的调度兼容性。

真实模型验收顺序：

1. **GLM BF16 单请求**：确认零生成 token、各层 post-residual、normalized final、token ID/长度、store key 和完成后可读性。
2. **BF16 多请求，再 CP/TP**：不同长度、padding 恢复、顺序、rank 唯一发布；CP 关闭 layer micro-batch。补测重复 ID、store 失败、异步错误及普通生成/MTP 无回归。
3. **FP8**：核对 scale/layout，匹配反量化后与 BF16 比较，再做多请求/CP/TP；容差根据真实训练要求确认，不预填精度结论。
4. **draft forward/loss**：真实 tokenizer、labels/mask 对齐后，验证有限 loss、有效 token 数和训练所需 backward/梯度，再扩大训练。

每次 E2E 切模型前必须先 warm page cache，并等成功后再启动。下列是**非直接执行的模板**，先确认目标机器上脚本存在并替换真实模型目录；本次未执行模型预热：

```text
/opt/conda310/bin/python3 /home/luhuanbing.lhb/RTP-LLM/local-lab/warm_model_page_cache.py <实际模型目录>
```

所有 E2E 日志/数据、临时实验和第三方构建产物放在 `/home/luhuanbing.lhb/RTP-LLM/local-lab`。Python/包管理使用 `/opt/conda310/bin/python3` / `/opt/conda310/bin/python3 -m pip`，安装版本待确认。

## 本机验证边界

此前在 PPU 主机上尝试过隔离的 CUDA13/SM10x 编译，但未产出 wheel，未证明完整编译通过；该尝试已停止，临时构建目录已清理。未运行 GPU 程序、真实模型、分布式传输或训练，也未改动共享 PPU/Python 环境。

测试已移除假 torch、自制 Tensor、AST 提取和源码字符串断言，改为直接调用真实实现：

- `hidden_state_capture_test.py`：真实 torch Tensor、CaptureContext、GenericMoe forward/capture hooks 与原生绑定；加速器 decoder 构造、缓存与诊断 I/O 保留最小边界替身。
- `pywrapped_model_cache_store_integration_test.py`：真实 C++ fixture、publisher、PyWrappedModel 和 FP8 quantizer；仅外部存储、通信及故障注入边界保留替身。
- 配置与 RPC 回归：真实配置、protobuf、torch 和 ops 导入；transport 使用确定性替身以覆盖取消、错误及资源释放。

**重写后的测试尚未运行，先前 30 项 mock／源码契约检查的通过结果不能作为当前测试的通过证据。** 本机缺少对应 torch/TorchSpec/native 运行环境，只完成了静态审查、语法和格式检查；请在目标机器执行上面的 Bazel targets。

实现依据：[`model_config.py`](../rtp_llm/config/model_config.py)、[`generate_config.py`](../rtp_llm/config/generate_config.py)、[`CaptureContext`](../rtp_llm/models_py/model_desc/hidden_state_capture.py)、[`generic_moe.py`](../rtp_llm/models_py/model_desc/generic_moe.py)、[`HiddenStateCapturePublisher.cc`](../rtp_llm/cpp/models/HiddenStateCapturePublisher.cc)、[`PyWrappedModel.cc`](../rtp_llm/cpp/models/PyWrappedModel.cc)、[`model_rpc_client.py`](../rtp_llm/cpp/model_rpc/model_rpc_client.py)。
