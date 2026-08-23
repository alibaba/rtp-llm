# Hidden-State 输出长度预测器（Length Predictor）集成方案

> 日期：2026-08-23
> 状态：v2 已实现（observe-only，双栈锚点）。C++ 编译通过（cuda12_9 与 cuda13）、
> 单测 9/9 通过、真权重（DeepSeek-V4-Pro seed42）已导出并完成离线数值对拍；
> 引擎内 smoke 待 8 卡环境（本机 4 卡拓扑不被 DSv4 支持，见 §9）。
> 来源研究：decode-predict 仓库 `Validation-Calibrated Prefill→History fusion`
>（离线 Test：fixed-age pairwise 63.45% → 79.22%，MAE 377.7 → 210.1 token）
>
> 离线对拍实测（100 条 test 轨迹，同一批缓存 hidden）：
> `anchor_total` median |Δ| = 0.95 token、`total_prediction` median |Δ| = 1.70 token。

## 1. 背景与目标

服务系统若能在请求运行中实时估计"剩余输出长度"，可用于长短分桶路由、
SRPT 类调度、KV 预算规划等。decode-predict 项目已离线验证：主模型
hidden state 中存在可提取的长度信号，且最优方案不是学习式融合，而是
**两个冻结预测器 + 一条 validation 锁定的固定 age 切换曲线**。

本方案把该最终推荐版实现进 rtp-llm 引擎，第一期目标：

1. **只观测不干预**：预测值写入 `AuxInfo.predicted_remaining_len`，不接调度器；
2. **零侵入开关**：环境变量控制，未启用时完全零开销；
3. **权重可配**：权重包从磁盘加载。DeepSeek-V4-Pro 的真权重已训练并导出
   （seed42，见 §5/§6）；random-init 包仍可用于无产物时的链路联调。

## 2. 在线算法

每个请求维护 ~700B 持久状态：`anchor_total`（标量）、32 维 GRU state、
`last_obs_step`、`predicted_total`。

**两个冻结模型，各有自己的 encoder + head**（这是与部署产物一致的结构，
见 §5 权重合同）：

- **anchor 栈**：prefill 专用探针（decode-predict 的 `prefill_initial` /
  `prefill_oof_*`）。只在 t=0 用一次。
- **history 栈**：`hidden_conditioned_full_history`，内含冻结的
  `point_model`（`universal_current`）作为 encoder+head，外加
  adapter / GRU / FiLM 调制。decode 步全部走它。

两栈形状完全相同、权重不同。若权重包未携带 anchor 栈（`anchor_*` 张量缺失），
anchor 自动回退复用 history 栈，保持对单栈包的兼容。

```text
Prefill 结束（t=0，最后一个 prompt token 的 final hidden，[1,H]）:
  hidden -> [anchor 栈] LayerNorm + Linear(H->128) + GELU
         -> + anchor_time_encoder(phi(0)) -> anchor fusion head -> 64-bin 分布
         -> anchor_total = expm1( Σ p_i · c_i )        (c_i 为 log1p bin center)
  hidden -> [history 栈] encoder -> feat128
         -> adapter(128->32) + [Δt=0 编码] -> GRU 写入   (先预测、后写入)
  注：GRU 始终用 history 栈的特征播种，使 decode 观察留在同一特征空间。

每个 decode step t（t = 已生成 token 数，全部走 history 栈）:
  观察网格 (t-1) % 4 == 0:   feat128 -> adapter -> GRU 更新（Δt 编码 [Δt/4, log1p(Δt)/log1p(4)]）
  预测网格 t % 20 == 0:      用"本步之前"的 GRU state 走 FiLM:
                              scale = 1 + 0.5·tanh(·), shift = 0.5·tanh(·)
                              modulated = feat·scale + shift
                              -> 冻结 head -> remaining 分布 -> history_total = remaining + t
                              alpha(t) = t<=100 ? 0 : 1 - 2^(-(t-100)/40)
                              predicted_total = expm1((1-α)·log1p(anchor) + α·log1p(history_total))
  同一步既是观察又是预测时，严格"先预测、后写入"（因果保证）。

任意时刻读数: predicted_remaining = predicted_total - 已生成 token 数（countdown）
```

时间编码 `phi(t) = [t/T_cap, log1p(t)/log1p(T_cap)]`，`T_cap` 来自权重包
config，超界不截断。所有组件冻结，无学习式 gate。

> 为什么必须分两栈：离线产物 `predictions_test.pt` 的 `anchor_total_prediction_raw`
> 恒等于 `prefill_total_prediction_raw`，且其 `prefill_checkpoint_sha256`
> 指向 prefill 专用 checkpoint。用 history 栈内嵌的 `point_model`（`universal_current`）
> 去算 anchor 会偏离 median 63 token；换用 prefill 专用栈后降到 median 0.95 token。

## 3. 集成架构

### 3.0 两条核心性质

这套集成的全部设计取舍都服务于一个目标：**observe-only 的功能绝不允许影响引擎**。
具体落成两条可验证的性质。

**性质一：引擎关键路径零 CUDA 调用**

dispatch 线程是每个 step 必经的引擎路径。它在这里只做三件事：维度/行合法性校验、
收集网格上的行、一次入队。**不调用任何 CUDA API、不分配显存、不发射 kernel。**

```text
// dispatch 线程（引擎关键路径）
entries = 遍历流收集网格上的 {stream, row, t}     // wantsStep 过滤
submitStep(hidden, entries):
    校验 + push(有界队列, {hidden 引用, entries}) + notify
// 结束 —— 零 CUDA API、零显存分配、零 kernel launch
```

**性质二：引擎关键路径零同步**

所有 GPU 提交和等待都发生在 worker 自己的私有 stream 上。worker 只
`cudaStreamSynchronize` 自己那条 stream，**主 forward stream 既不排队也不等待**，
预测 kernel 反而可以去填空闲 SM。

```text
// worker 线程（旁路，与主 forward 并行）
在私有 stream:  encoder kernel  →  async D2H
cudaStreamSynchronize(私有 stream)      // ← 只阻塞 worker 自己
```

再加一条兜底：队列有界，积压即整包丢弃并计数告警。**即使 worker 完全跟不上，
关键路径也不会被拖慢一微秒**——代价只是丢掉若干观察/预测点。

实测量级（B=32，B300）：

| 位置 | 开销 |
|---|---|
| dispatch 线程（关键路径） | 校验 + 入队，**~1 µs，零 CUDA** |
| worker 私有 stream：encoder kernel | 0.075 ms（与主 forward 重叠） |
| worker 私有 stream：D2H `[32,128]` | 0.004 ms（16 KB，延迟受限） |
| worker 线程：CPU 数学 | 观察 0.006 ms/行、预测 0.036 ms/行 |

GPU encoder 的耗时**几乎与 batch 无关**（B=1 → 0.039 ms，B=256 → 0.083 ms，
latency-bound），因此每请求的边际成本随 batch 增大趋近于 0。

#### 这两条性质由接口设计保证

不是靠"小心写"，而是接口本身让关键路径**没有机会**碰 CUDA 或等待。

保证零 CUDA：

| 接口 | 设计选择 | 效果 |
|---|---|---|
| `static LengthPredictor* instance()` | 可空裸指针，非工厂/非虚 | 未启用时调用点仅一次判空 |
| `submitStep(const Tensor&, vector<Entry>)` | hidden 按 const 引用传入、只存引用；返回 void | 不拷贝、不分配显存、不发 kernel；靠 tensor 引用计数钉住显存 |
| `bool wantsStep(int64_t) const` | 网格判据暴露给调用方，纯整数、可内联 | entries 规模随网格行数而非 batch |
| `Entry.state` 为裸指针 | 状态由 `GenerateStream` 持有，predictor 不拥有 | 关键路径无 map 查找、无分配 |
| `Entry.keepalive` 为 `shared_ptr<void>` | 类型擦除持有 `GenerateStreamPtr` | 不依赖 stream 库（无循环依赖），又保证对象存活 |
| `ensureDeviceWeights()` private、worker 调用 | 权重上传 GPU 懒加载且在旁路 | 首包 H2D 也不落关键路径 |

保证零同步：

| 接口 | 设计选择 | 效果 |
|---|---|---|
| `submitStep` 契约：有界队列 + 淘汰最旧 | 明示"异步、非阻塞、可丢弃" | 队列满不阻塞、不反压，只 `dropped++` |
| `worker_stream_`（私有 pool stream） | predictor 自持一条 stream，从不用 current stream | `cudaStreamSynchronize` 只作用于自身 |
| **`submitStep` 前置条件写进接口** | 要求调用方在其 token D2H 同步之后调用 | **同步继承自引擎**，因此无需 `cudaEventRecord`/`StreamWaitEvent` 做跨 stream 排序 |
| `predicted_total` 为 `atomic<double>` | 唯一跨线程字段，relaxed load/store | 结果发布无需 mutex、无需 CUDA event |
| `predictedRemainingLen()` 读时算 countdown | 返回 `total - generated` 而非每步更新的值 | 写频率（每 20 token）与读频率（每步）解耦 |
| 不提供 completion callback、不暴露 CUDA event | 刻意省略 | 避免 worker 回调进引擎、避免引入顺序依赖 |

一句话：`submitStep` 被设计成**只交接引用、不做工作**，加上**同步继承自调用方**这条
契约，两条性质是必然结果，而非需要维护的约定。

### 3.1 数据流（复用现有管道，无新增模型 forward）

`GptModelOutputs.hidden_states` 是 lm_head 的输入行（过 final-layernorm、
按 `lm_output_indexes` 选出的每 batch 行最后一个 token hidden），**恒有**：

- context 步：该行即最后一个 prompt token hidden → 锚点输入；
- decode 步：该行即当前 token hidden → 观察/预测输入。

因此无需改 `NormalModelInputGatherer`、无需 `need_all_hidden_states`。

### 3.2 执行模型（v3：关键路径零 CUDA 调用，worker 独立 stream 全权负责 GPU）

```text
dispatch 线程（每 step 一次，~1µs，无任何 CUDA API）:
  NormalOutputDispatcher::dispatch
    └─ LengthPredictor::submitStep(model_output.hidden_states [B,7168], entries)
         仅做：维度/行合法性校验 → {hidden 引用, entries} 入有界队列（≥4 包积压即丢）
         前置条件：钩子位于本步 token D2H 同步之后 —— 此刻产生 hidden 的
         kernel 已全部完成，worker 才能从另一条 stream 直接读它而无需跨
         stream event；packet 持有的 tensor 引用防止 caching allocator 复用该显存

CPU worker 线程（持有独立 pool CUDA stream，全部 GPU 提交和条件逻辑在这）:
  1. 在自己的 stream 上发射手写融合 kernel（standardize + GEMM(W′) + bias′ + GELU(erf)，
     整批全算不 gather）→ cudaMemcpyAsync 到 pinned ring 槽位
  1b. 若本包含 t==0 行且权重包带 anchor 栈：用 anchor 栈权重再发射同一 kernel
     一次 → 写入该槽位的第二块 pinned buffer。prefill 步每请求最多一行，
     普通 decode 步完全跳过这次额外 encode
  2. cudaStreamSynchronize（只阻塞 worker 自己；与主 stream 完全隔离，
     预测 kernel 可与下一步 forward 在空闲 SM 上并行）
  3. 逐行按该流的 t 分发（手写 fp32 数学，不经 torch CPU 算子）:
       t == 0          → 锚点：anchor_head(anchor_feat, phi(0)) → anchor_total；
                          随后把 history 栈的 feat 写入 GRU
       (t-1) % 4 == 0  → GRU 观察更新
       t % 20 == 0     → FiLM + history head + α(t) 融合 → 原子写 predicted_total
       其余             → 丢弃该行
GenerateStream::length_predictor_state_             per-request 状态（CPU 原生）
GenerateStream::predictedRemainingLen()             countdown 读数（读 atomic）
NormalGenerateStream::updateOutput                  写入 AuxInfo
```

关键设计点：

- **γ/β 折叠**：加载权重时执行 `W′ = W·diag(γ)`、`b′ = b + W·β`，运行时
  LayerNorm 退化为纯标准化，融合进单 kernel；
- **权重转置**：W′ 存 `[7168,128]`，kernel 的 rank-1 累加循环内 128 线程
  读连续 512B，完全 coalesced；W′ 仅 3.6MB，跨 block 全部 L2 命中；
- **GELU 用 erf 精确式**（`0.5x(1+erf(x/√2))`），与训练时
  `torch.nn.GELU()` 默认行为逐位对齐；CPU 侧同样用 `std::erf`；
- **滞后一拍语义**：预测值经 worker 异步写回，晚 1~2 个 step 才在
  aux_info 可见。对每 20 token 刷新一次的 countdown 无感知影响；
- **生命周期与并发**：packet 持 `GenerateStreamPtr` 保证 stream 存活；
  `predicted_total` 为 atomic（dispatch/输出线程读、worker 写）；其余状态
  字段仅 worker 单线程访问；单 worker + FIFO 保证同流观察严格有序；
- **永不反压**：队列有界，积压即整包丢弃并计数告警——observe-only 功能
  不允许拖慢引擎；
- **CPU-only 构建回退**：无 CUDA 时 kernel 路径退化为等价 torch 算子实现。

预测器为进程级单例（`LengthPredictor::instance()`），由
`RTP_LLM_LENGTH_PREDICTOR_CHECKPOINT` 环境变量控制：未设置返回
nullptr（调用点一次指针判断，零开销）；加载失败打 ERROR 日志并禁用，
不影响服务启动。

### 3.3 开销预算（v3）

| 位置 | 成本 | 说明 |
|---|---|---|
| dispatch 线程（关键路径） | 校验 + 1 次 mutex 入队 + cv notify，~1µs | **零 CUDA API、零同步** |
| GPU（worker 私有 stream） | 1 kernel + 1 DMA：实测 B=32 为 0.075 ms + 0.004 ms | 不占主 stream 队列，与下一步 forward 并行；耗时近乎与 B 无关 |
| worker 线程 CPU 数学 | 实测 0.006 ms/行（观察）、0.036 ms/行（预测） | 后台单核百分之几占用 |
| 显存 | W′ 3.6MB（双栈 7.2MB）+ 4 槽 × [B,128] scratch（双栈时 prefill 步再加一块） | 常驻 |
| 内存 | pinned ring（4 槽 × B×512B）+ 每请求 ~200B 状态 | 常驻 |

## 4. 改动清单

### rtp-llm

| 文件 | 改动 |
|---|---|
| `rtp_llm/cpp/length_predictor/LengthPredictorState.h` | 新增：per-request 状态 struct（CPU 原生 float 数组 + atomic predicted_total） |
| `rtp_llm/cpp/length_predictor/LengthPredictor.{h,cc}` | 新增：权重加载（含 γ/β 折叠与转置）、submitStep 异步提交、worker 线程、手写 CPU 数学、α 融合。v2：可选 anchor 栈（`HeadWeights` 选择器、第二组 encoder/head、slot 内第二块 pinned buffer、`hasAnchorModel()`），缺失时回退单栈 |
| `rtp_llm/cpp/length_predictor/kernels/length_encoder_kernel.{h,cu}` | 新增：手写融合 kernel（standardize+GEMM+bias+GELU），bf16/fp16/fp32 输入 |
| `rtp_llm/cpp/length_predictor/BUILD` | 新增：state（header-only）、kernel（cuda）、length_predictor 三个目标 |
| `rtp_llm/cpp/length_predictor/test/LengthPredictorTest.cc` | 新增：单测（见 §6） |
| `rtp_llm/cpp/length_predictor/test/BUILD` | 新增 |
| `rtp_llm/cpp/engine_base/stream/GenerateStream.h` | 状态成员 + `lengthPredictorState()` + `predictedRemainingLen()` |
| `rtp_llm/cpp/engine_base/stream/GenerateTypes.h` | `AuxInfo.predicted_remaining_len`（float，负值=未启用） |
| `rtp_llm/cpp/engine_base/stream/BUILD` | stream 依赖 `length_predictor_state` |
| `rtp_llm/cpp/normal_engine/NormalOutputDispatcher.cc` | dispatch 批级钩子：遍历流时收集 {stream, row, t}，循环后一次 `submitStep` |
| `rtp_llm/cpp/normal_engine/NormalGenerateStream.cc` | aux_info 填充 |
| `rtp_llm/cpp/normal_engine/BUILD` | 依赖 `length_predictor` |
| `rtp_llm/utils/ckpt_file_info.py` | v2：`ROCM_COPY_OUT` 增加 `RTP_LLM_CKPT_COPY_OUT=1` opt-in（默认不变）。见 §9 host 内存 |

### decode-predict

| 文件 | 改动 |
|---|---|
| `src/export_online_predictor.py` | 新增：权重导出。`--history-checkpoint` 读 `hidden_conditioned_full_history`；`--prefill-checkpoint` 可选，带上则额外导出 `anchor_*` 栈；`--calibration` 读 `calibration.json` 自动取锁定的 α 曲线参数。导出后回读校验张量数/shape/单调性/有限性 |
| `src/offline_parity.py` | 新增：离线对拍。numpy 忠实复刻 C++ 算法（只读导出的权重包），跑同一批缓存 hidden，与 `predictions_test.pt` 的 `anchor_total_prediction_raw` / `candidate_total_prediction_raw` / `total_prediction_raw` 比对 |

## 5. 权重包合同

TorchScript 容器（`torch.jit.script(Module).save()`），C++ 侧经
`torch::jit::load` + `named_buffers()` 读取扁平命名 buffer：

- **history 栈权重（21 个 tensor，必需）**：`encoder_ln_{weight,bias}`、
  `encoder_linear_{weight,bias}` `[128,H]`、`time_linear_*` `[16,2]`、
  `fusion1_*` `[128,144]`、`fusion2_*` `[64,128]`、`adapter_*` `[32,128]`、
  `gru_{weight,bias}_{ih,hh}` `[96,34]/[96,32]`、`modulator1_*` `[42,32]`、
  `modulator2_*` `[256,42]`、`bin_centers` `[64]`（log1p 空间、严格递增）；
- **anchor 栈权重（11 个 tensor，可选）**：`anchor_encoder_ln_{weight,bias}`、
  `anchor_encoder_linear_{weight,bias}`、`anchor_time_linear_*`、
  `anchor_fusion1_*`、`anchor_fusion2_*`、`anchor_bin_centers`
  —— 形状与 history 栈同名张量逐一对应。以
  `anchor_encoder_linear_weight` 是否存在作为"带 anchor 栈"的判据；
  缺失则 anchor 回退复用 history 栈；
- **config（14 个标量 buffer，`config_` 前缀）**：`hidden_dim`、`feature_dim`、
  `state_dim`、`adapter_dim`、`time_dim`、`num_bins`、`t_cap`、`scale_limit`、
  `shift_limit`、`history_stride`、`predict_stride`、`hard_anchor_until`、
  `half_life_tokens`、`max_history_weight`。

因此合法的包是 **21+14=35** 个 buffer（单栈）或 **32+14=46** 个（双栈）。

C++ 加载时逐 tensor 校验 shape/有限性/单调性，任何缺失或错形直接拒载。

## 6. 使用与验证

### 使用

```bash
# 真权重导出（DeepSeek-V4-Pro，seed42，双栈）
D=<dataset>/03-weights-and-predictions/prefill-decode-full-stride4-ablation/disjoint_fold0
python decode-predict/src/export_online_predictor.py \
    --history-checkpoint $D/new_full_history_seed42/checkpoint_best.pt \
    --prefill-checkpoint $D/prefill_oof_seed42_fold0/checkpoint_best.pt \
    --calibration       $D/validation_calibrated_v6/calibration.json \
    --output /path/length_predictor_dsv4pro_seed42_2stack.pt
# 期望输出：46 buffers (32 weights + 14 config), anchor stack: prefill-specialised

# 启动引擎
export RTP_LLM_LENGTH_PREDICTOR_CHECKPOINT=/path/length_predictor_dsv4pro_seed42_2stack.pt
# 请求带 aux_info=true，读 aux_info.predicted_remaining_len
```

启动日志会打出 `length predictor enabled from ... anchor_stack=prefill-specialised`
（或 `shared-with-history`），可据此确认加载的是哪种包。

### 已完成验证

- **编译**：`--config=cuda12_9` 与 `--config=cuda13` 均通过（cuda13 见 §9 前置条件）；
- **单测 9/9 通过**（`//rtp_llm/cpp/length_predictor/test:length_predictor_test`）：
  锚定窗精确 countdown、α 曲线三点值（`alpha(140)=0.5 / alpha(180)=0.75 /
  alpha(260)=0.9375`）、log1p 融合端点精确性、因果性（同一步"先预测后写入"，
  手工用旧 state 复现预测值比对）、手写 CPU GRU/FiLM/head 与 torch 参考实现
  数值对齐、融合 kernel 与 torch `layer_norm+linear+gelu` 数值对齐、
  off-grid 行丢弃、坏权重拒载、worker 异步写回与队列丢弃路径；
- **导出**：权重包往返读取正确（46 buffers，shape/单调性/有限性校验通过）；
  history 栈 SHA 与部署 manifest 的 `unanchored_checkpoint_sha256` 一致，
  anchor 栈 SHA 与 `prefill_checkpoint_sha256` 一致；
- **离线数值对拍**（`offline_parity.py`，100 条 test 轨迹、同一批缓存 hidden）：

  | 量 | median \|Δ\| | 说明 |
  |---|---|---|
  | `anchor_total` | **0.95 token** | 双栈修正前为 230.9 |
  | `total_prediction`（实际发布值） | **1.70 token** | 双栈修正前为 14.0 |
  | history candidate | 3.70 token | prefill 播种；不播种为 30.5，佐证 GRU 播种方式正确 |

### 待验证（需 8 卡环境）

引擎级 smoke：起服务发长文请求，确认 `predicted_remaining_len` 在 t≤100 呈
斜率 -1 的 countdown、t>100 后逐步偏离锚点、每 20 token 刷新一次（允许滞后
1~2 step）；随后做 predictor 开/关的吞吐 / TTFT / ITL 对比，并确认
`droppedPackets()` ≈ 0。

本机 4 卡无法完成：DSv4 在 TP=4 时每卡 query head = 32，MLA kernel 报
`Unsupported h_q: 32`（`MLA_OPS_TYPE=FLASH_INFER` 亦无效）；改纯 TP
（`EP_SIZE=1`）虽绕过 MoE 的 Mega 要求，但权重需 264.93 GiB/卡 > 267.69 GiB
可用（限制 KV cache 也不够）。线上为 8 卡拓扑，需在该环境补做。

## 7. 边界与风险

1. **权重有效性**：预测器与主模型强绑定。**DeepSeek-V4-Pro（hidden 7168、
   final 层）的真权重已导出并离线验证**（§6）；**V4 Flash 必须用
   decode-predict 管道重新采数据训练**，random-init 包只验证链路。
2. **两栈必须配对**：anchor 栈与 history 栈要来自同一 seed、同一
   calibration 运行。混用不同 seed 会让 α 融合的两端不在同一标定下，
   离线对拍会立刻暴露（anchor 偏差跳到百 token 量级）。
3. **生效范围**：仅 NormalEngine、非 beam、单 batch 流。
   MTP/speculative 走 `MtpBatchStreamProcessor`，不经过本钩子（静默不生效）；
   PD 分离 decode 角色无本地 prefill hidden，锚点不建立，该请求预测器关闭；
   `num_return_sequences>1` 跳过。
4. **滞后与丢弃语义**：预测值晚 1~2 step 可见；worker 积压时整包丢弃，
   丢的是若干观察/预测点，GRU 用 Δt 编码天然容忍非均匀间隔，但持续丢弃
   会降低预测质量——按丢弃计数告警。
5. **精度边界**（继承离线结论）：fixed-age 排序好 ≠ 调度收益；
   接调度器前必须先做真实 queue replay（decode-predict HANDOFF §9 红线）。
6. **hidden 语义**：取的是过 final-layernorm 的 lm_head 输入行；
   训练数据采集必须对齐同一层、同一归一化位置，否则分布漂移。
7. aux_info 的 proto 透传（`QueryConverter` → 客户端）未实现，
   当前引擎内与日志可见。

## 8. 后续计划

1. **8 卡环境补做引擎 smoke + 性能对比**（§6 待验证），这是唯一缺口；
2. proto/Python 侧透传 `predicted_remaining_len`，接 kmonitor 指标；
3. 融合 kernel 进 CUDA graph（shape 已固定为 [B,7168]，无阻碍）；
4. queue replay 验证后，评估接 scheduler（长短分桶 / SRPT）；
5. V4 Flash 数据采集 + 预测器训练（decode-predict 已有完整管道），
   用 `export_online_predictor.py` 打真权重包。

## 9. 环境前置条件（本次调试记录）

predictor 本身对环境无特殊要求；以下是把 **DSv4 Pro 跑起来** 所需的前置条件，
在 Blackwell（B300 / sm103）+ CUDA 12.9 镜像上逐个踩到，记录以便复现。

### 9.1 DSv4 MoE 需要 cuda13 工具链

`dsv4/moe/strategies/base.py` 对 `ep_size > 1` **强制要求 MegaMoEStrategy**，
无 env 可绕过（显式 `DSV4_MOE_STRATEGY` 也会在另一处 raise）。而
`mega_buf.py:_mega_moe_unavailable_reason()` 的可用性条件是：
`deep_gemm.fp8_fp4_mega_moe` 存在 + `torch.distributed` 已初始化 +
world_size > 1 + **device capability ≥ sm100**。

- 硬件侧 sm103 满足（注意 `nvidia-smi` 可能把设备名/`compute_cap` 报成
  `L20D` / `8.9`，**以 `torch.cuda.get_device_capability()` 为准**，实测 `(10, 3)`）；
- 但 `--config=cuda12_9` pin 的 `deep_gemm 2.1.1+local` **不含**该 kernel；
  含它的是 cuda13 配置 pin 的 `deep_gemm 2.5.0+8a4dfba`（需 `libcudart.so.13`，
  无法在 12.9 运行时里混用）。

结论：**必须走 `--config=cuda13`**（torch 2.11+cu130 + deep_gemm 2.5.0）。

### 9.2 用 cuda13 配置构建的三个坑

1. **本地需装 CUDA 13.2 toolkit**。`build:cuda13_base` 只设
   `TF_CUDA_VERSION=13.2`，toolkit 路径仍继承 `build:cuda` 的
   `/usr/local/cuda/`。可装到独立前缀并用 flag 覆盖，不必改系统符号链接：

   ```bash
   sudo ./cuda_13.2.0_595.45.04_linux.run --silent --toolkit \
        --toolkitpath=/usr/local/cuda-13.2 --no-man-page --override
   ```

   runfile **不含 cuDNN/NCCL**，需自行放入该目录（本次取
   `nvidia-cudnn-cu13==9.19.0.56` / `nvidia-nccl-cu13==2.28.9` 两个 wheel 的
   `lib/`+`include/` 复制进 `targets/x86_64-linux/`，并补 `.so` 软链）。

2. **CUDA 13 把 CCCL 头文件移到 `include/cccl/`**，`cuda/std/*` 不再直接位于
   `include/` 下，宿主编译会报 `fatal error: cuda/std/utility`。加 include
   路径即可，无需改源码：

   ```
   --copt=-isystem --copt=bazel-out/k8-opt/bin/external/local_config_cuda/cuda/cuda/include/cccl
   （--host_copt 同）
   ```

   另外：换 config 前务必 `bazelisk clean`，否则 12.9 遗留的
   `cuda/include/cuda/std/**` 会被 nvcc 找到并触发 strict-deps
   "undeclared inclusion" 报错。

3. **B300 需补 sm103**。x86 `build:cuda13` 的
   `TF_CUDA_COMPUTE_CAPABILITIES` 只列到 `10.0`（ARM 的 `cuda13_arm` 才有
   `10.0,10.3`），本次构建用
   `--action_env TF_CUDA_COMPUTE_CAPABILITIES=9.0,10.0,10.3`（+`--host_action_env`）。

完整构建命令见 git 历史；`length_predictor_test` 在 cuda13 下链接会因
torch 2.11 `libtorch_python.so` 的 `c10d::nvshmem_extension::*` 未定义符号失败，
单测已在 cuda12_9 下跑过，构建时可只 build `rtp_llm:rtp_llm`。

### 9.3 deep_gemm JIT 需要 gcc ≥ 11（本次用 12）

deep_gemm 2.5.0 在运行时用 nvcc **JIT 编译** kernel。
`sm100_fp8_fp4_mega_moe.cuh` 使用了 `float` 非类型模板参数（C++20 特性），
而系统 gcc 10.2 不支持（`'float' is not a valid type for a template
non-type parameter`），导致 nvcc 前端报
`floating-point template parameter is nonstandard`（硬错误，不可 suppress，
且与 `-std=c++20` 无关）。依赖注释也写明 wheel 是用 gcc-toolset-12 构建的。

修法：装一个较新 gcc，并让 deep_gemm 的 JIT 用它作宿主编译器。
`DG_JIT_NVCC_COMPILER` 只接受 nvcc 路径，故用一层 shim 注入 `-ccbin`：

```bash
# shim: exec /usr/local/cuda-13.2/bin/nvcc -ccbin <gcc12>/x86_64-conda-linux-gnu-g++ "$@"
export CUDA_HOME=/usr/local/cuda-13.2          # deep_gemm 据此找 nvcc
export DG_JIT_NVCC_COMPILER=/path/to/nvcc-shim
```

验证方式：起服务后 `find <DG_JIT_CACHE_DIR> -name '*.cubin' | wc -l` 应 > 0
且日志无 `NVCC compilation failed`（本次 25 个 cubin 全部编译成功）。

### 9.4 权重加载的 host 内存（memory cgroup OOM）

`ckpt_file_info.py` 的 `ROCM_COPY_OUT` 原为 `torch.version.hip is not None`，
即 CUDA 上恒 false；而 `CkptDatabase._recycle_handles` 依赖它，于是 CUDA 上
**mmap 句柄回收被永久关闭**（日志 `recycle_handles=False (asked=True,
copy_out=False)`）。权重拷进 GPU 后 host 端 mmap 页从不释放，page cache
单调涨到整个 checkpoint（671GB 模型实测每 rank `file-rss ≈ 170GB`，
4 rank 撞爆 pod 的 memory cgroup，dmesg 报 `CONSTRAINT_MEMCG`）。

已加 opt-in 开关（默认行为不变）：

```bash
export RTP_LLM_CKPT_COPY_OUT=1   # 使 recycle_handles=True，page cache 受控
```

开启后 671GB 权重可完整加载（实测 ~170GB/卡显存，无 host OOM）。

### 9.5 4 卡拓扑不可用

见 §6"待验证"：TP=4 → h_q=32，MLA kernel 不支持；纯 TP 显存不够。
**线上为 8 卡拓扑**，引擎级验证需在该环境进行。
