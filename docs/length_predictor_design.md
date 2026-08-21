# Hidden-State 输出长度预测器（Length Predictor）集成方案

> 日期：2026-08-18
> 状态：v1 已实现（observe-only），C++ 编译验证待构建环境
> 来源研究：decode-predict 仓库 `Validation-Calibrated Prefill→History fusion`
>（离线 Test：fixed-age pairwise 63.45% → 79.22%，MAE 377.7 → 210.1 token）

## 1. 背景与目标

服务系统若能在请求运行中实时估计"剩余输出长度"，可用于长短分桶路由、
SRPT 类调度、KV 预算规划等。decode-predict 项目已离线验证：主模型
hidden state 中存在可提取的长度信号，且最优方案不是学习式融合，而是
**两个冻结预测器 + 一条 validation 锁定的固定 age 切换曲线**。

本方案把该最终推荐版实现进 rtp-llm 引擎，第一期目标：

1. **只观测不干预**：预测值写入 `AuxInfo.predicted_remaining_len`，不接调度器；
2. **零侵入开关**：环境变量控制，未启用时完全零开销；
3. **权重可配**：主模型（DeepSeek V4 Flash）的预测器权重尚未训练，
   框架先行，权重包从磁盘加载，支持 random-init 联调。

## 2. 在线算法

每个请求维护 ~700B 持久状态：`anchor_total`（标量）、32 维 GRU state、
`last_obs_step`、`predicted_total`。

```text
Prefill 结束（t=0，最后一个 prompt token 的 final hidden，[1,H]）:
  hidden -> LayerNorm(H) + Linear(H->128) + GELU        (current encoder)
  feat128 + time_encoder(phi(0)) -> fusion head -> 64-bin 分布
  anchor_total = expm1( Σ p_i · c_i )                    (c_i 为 log1p bin center)
  feat128 -> adapter(128->32) + [Δt=0 编码] -> GRU 写入   (先预测、后写入)

每个 decode step t（t = 已生成 token 数）:
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

## 3. 集成架构

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
  2. cudaStreamSynchronize（只阻塞 worker 自己；与主 stream 完全隔离，
     预测 kernel 可与下一步 forward 在空闲 SM 上并行）
  3. 逐行按该流的 t 分发（手写 fp32 数学，不经 torch CPU 算子）:
       t == 0          → 锚点：head(feat, phi(0)) → anchor_total；feat 写入 GRU
       (t-1) % 4 == 0  → GRU 观察更新
       t % 20 == 0     → FiLM + head + α(t) 融合 → 原子写 predicted_total
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
| GPU | 每 step 1 kernel + 1 DMA，在 worker 私有 stream 上 | 不占主 stream 队列，与下一步 forward 并行 |
| worker 线程 | streamSync 等待 + ~0.3B 行 CPU 数学（B=256 时 ~1.3M MAC） | 后台单核百分之几占用 |
| 显存 | W′ 3.6MB + 4 槽 × [B,128] scratch | 常驻 |
| 内存 | pinned ring（4 槽 × B×512B）+ 每请求 ~200B 状态 | 常驻 |

## 4. 改动清单

### rtp-llm

| 文件 | 改动 |
|---|---|
| `rtp_llm/cpp/length_predictor/LengthPredictorState.h` | 新增：per-request 状态 struct（CPU 原生 float 数组 + atomic predicted_total） |
| `rtp_llm/cpp/length_predictor/LengthPredictor.{h,cc}` | 新增：权重加载（含 γ/β 折叠与转置）、submitStep 异步提交、worker 线程、手写 CPU 数学、α 融合 |
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

### decode-predict

| 文件 | 改动 |
|---|---|
| `src/export_online_predictor.py` | 新增：权重导出。真权重模式读 point/history checkpoint + `calibration.json`（自动取锁定的 α 曲线参数）；`--random-init` 模式生成同结构随机权重（FiLM 末层零初始化，行为精确退化为 Fresh Decode probe） |

## 5. 权重包合同

TorchScript 容器（`torch.jit.script(Module).save()`），C++ 侧经
`torch::jit::load` + `named_buffers()` 读取扁平命名 buffer：

- **权重（21 个 tensor）**：`encoder_ln_{weight,bias}`、
  `encoder_linear_{weight,bias}` `[128,H]`、`time_linear_*` `[16,2]`、
  `fusion1_*` `[128,144]`、`fusion2_*` `[64,128]`、`adapter_*` `[32,128]`、
  `gru_{weight,bias}_{ih,hh}` `[96,34]/[96,32]`、`modulator1_*` `[42,32]`、
  `modulator2_*` `[256,42]`、`bin_centers` `[64]`（log1p 空间、严格递增）；
- **config（14 个标量 buffer，`config_` 前缀）**：`hidden_dim`、`feature_dim`、
  `state_dim`、`adapter_dim`、`time_dim`、`num_bins`、`t_cap`、`scale_limit`、
  `shift_limit`、`history_stride`、`predict_stride`、`hard_anchor_until`、
  `half_life_tokens`、`max_history_weight`。

C++ 加载时逐 tensor 校验 shape/有限性/单调性，任何缺失或错形直接拒载。

## 6. 使用与验证

### 使用

```bash
# 联调（随机权重，无需训练产物）
python decode-predict/src/export_online_predictor.py --random-init \
    --hidden-dim <主模型 hidden dim> --t-cap 73581 --output /path/predictor.pt

# 启动引擎
export RTP_LLM_LENGTH_PREDICTOR_CHECKPOINT=/path/predictor.pt
# 请求带 aux_info=true，读 aux_info.predicted_remaining_len
```

### 已完成验证

- 导出脚本实测跑通；权重包往返读取正确（35 buffers、shape 校验通过）；
- Python 逐算子复现 C++ 全链路：`t<=100` 预测精确等于锚点 countdown，
  `alpha(140)=0.5 / alpha(180)=0.75 / alpha(260)=0.9375` 与离线报告一致，
  融合数值全程有限。

### 待验证（需构建环境）

```bash
bazelisk build //rtp_llm/cpp/length_predictor/...
bazelisk test  //rtp_llm/cpp/length_predictor/test:length_predictor_test
```

单测覆盖：锚定窗精确 countdown、α 曲线三点值、log1p 融合端点精确性、
因果性（同一步"先预测后写入"，手工用旧 state 复现预测值比对）、手写 CPU
GRU/FiLM/head 与 torch 参考实现数值对齐、融合 kernel 与 torch
`layer_norm+linear+gelu` 数值对齐（GPU 环境）、off-grid 行丢弃、坏权重拒载、
worker 异步写回与队列丢弃路径。

引擎级 smoke（编译通过后）：本地起服务发长文请求，确认
`predicted_remaining_len` 在 t≤100 呈斜率 -1 的 countdown、t>100 后
逐步偏离锚点、每 20 token 刷新一次（允许滞后 1~2 step）。

## 7. 边界与风险

1. **权重有效性**：预测器与主模型强绑定。现有训练权重基于
   DeepSeek V4 Pro（hidden 7168、final 层）；**V4 Flash 必须用
   decode-predict 管道重新采数据训练**，random-init 包只验证链路。
2. **生效范围**：仅 NormalEngine、非 beam、单 batch 流。
   MTP/speculative 走 `MtpBatchStreamProcessor`，不经过本钩子（静默不生效）；
   PD 分离 decode 角色无本地 prefill hidden，锚点不建立，该请求预测器关闭；
   `num_return_sequences>1` 跳过。
3. **滞后与丢弃语义**：预测值晚 1~2 step 可见；worker 积压时整包丢弃，
   丢的是若干观察/预测点，GRU 用 Δt 编码天然容忍非均匀间隔，但持续丢弃
   会降低预测质量——按丢弃计数告警。
4. **精度边界**（继承离线结论）：fixed-age 排序好 ≠ 调度收益；
   接调度器前必须先做真实 queue replay（decode-predict HANDOFF §9 红线）。
5. **hidden 语义**：取的是过 final-layernorm 的 lm_head 输入行；
   训练数据采集必须对齐同一层、同一归一化位置，否则分布漂移。
6. aux_info 的 proto 透传（`QueryConverter` → 客户端）未实现，
   当前引擎内与日志可见。

## 8. 后续计划

1. V4 Flash 数据采集 + 预测器训练（decode-predict 已有完整管道），
   用 `export_online_predictor.py` 打真权重包；
2. proto/Python 侧透传 `predicted_remaining_len`，接 kmonitor 指标；
3. 融合 kernel 进 CUDA graph（shape 已固定为 [B,7168]，无阻碍）；
4. queue replay 验证后，评估接 scheduler（长短分桶 / SRPT）。
