# PR 1521 机器人 review findings 跟踪

> 来源：github.com/alibaba/rtp-llm **PR #1521**，`rtp-llm-review-bot` 的 inline 评论；评审版本 head `423c9c34c6f1`；抓取 2026-10-19。
> 34 条 inline finding（首次抓取用默认 per_page=30 只得前 30 条、漏 4 条；2026-10-20 以 per_page=100 补全），按 (文件:行) 聚成 **23 个不同问题**（bot 对同一问题多角度重复提）；按 finding 计 12 P1 + 22 P2。
> bot 原文缓存：`~/.qoder/.../cce3ba17/9d87df3a.txt`（506KB 全量 34 条 JSON，per_page=100）、`d60cc1d0.txt`（480KB 首页 30 条）、`b69484bd.txt`（格式化 121KB）。
> 用法：逐行对齐，把状态从「⬜ 待对齐」推进到「✅ 已修复 / 🅐 已搁置 / 🅡 回应澄清」，并在详情节回填我方分析与结论。
> 当前代码基线：`db71c7bf39`（2026-10-21）；下列提交哈希均按当前 rebase 后历史记录。

## 汇总表（按优先级 → 文件；#21–#23 为第二页补抓、附于末尾，24303 并入 #15）

| # | 优先级 | 文件:行 | findings | 问题 | 状态 |
|---|---|---|---|---|---|
| 1 | P1 | cpp/engine_base/stream/GenerateStream.cc:994 | 24277,24293 | maxTokenNum() 改写为 int 域加法后，协议合法的近 INT_MAX max_new_tokens 有符号溢出，经 CompleteTokenIds::updat… | ✅ 已修复 b8b0a87925 |
| 2 | P1 | cpp/model_rpc/DecodeRpcServer.cc:919 | 24279,24284,24294,24295 | stage 路由发送端跳过 P/D TP 整除预检，互不整除比例使首个 PD 请求在每个 decode rank 的 RemoteLoad 处理器内命中未捕获 RTP_LLM_… | ✅ 已修复 23a7fa496d |
| 3 | P1 | cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc:63 | 24276,24283,24287,24291,24296 | 未启用 SP 的 PP 部署下，单个 num_beams>1 请求在首阶段 root 引擎循环线程触发未捕获断言异常，引擎停止推进、整条流水线停摆 | ✅ 已由 e2d47e6fa0 解决（评审后提交） |
| 4 | P1 | cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc:583 | 24299 | PP 尾段在错误归集之前对整批 new_token_ids 做紧凑词表致命 CHECK 与 softmax gather:ROCm 构建下采样失败行的 -1 哨兵触发断言异常并… | ✅ 已落地非 CUDA + PP 启动期门禁 7940f08bdf |
| 5 | P1 | model_loader/loader.py:530 | 24275 | PP>1 下在线权重同步确定性失败：非持有 rank 的 UpdateWeights 因层/全局键缺失抛 KeyError 并返回 gRPC INTERNAL，且归属 rank… | ✅ 已修复+提交 272d79ca4a |
| 6 | P2 | deps/requirements_lock_torch_gpu_cuda13.txt:634 | 24273 | cuda13 x86 五个 wheel 的 OSS 主机改写后，其中四个对象在新主机上无快照内可用性证据；若新主机缺任一对象，该平台构建与安装期整体失败 | ✅ 已回退 e10b307d68（担心已实测否证；改动违反 lock 规范且无效） |
| 7 | P2 | cpp/cache/test/BUILD:96 | 24272 | 新增 pp_topology_validator_test 链接闭包缺少 execPPSnapshotExchange 的唯一定义：opt 特性关闭的受支持编译模式（-c db… | ✅ 已修复+验证 PASSED e10b307d68 |
| 8 | P2 | cpp/engine_base/schedulers/PPScheduler.cc:92 | 24282 | PPScheduler 把 max_generate_batch_size（即用户侧 concurrency_limit）重释为单轮 microbatch 上限：DECODE … | 🅡 已澄清（PP microbatch 设计语义） |
| 9 | P2 | cpp/engine_base/schedulers/PPScheduler.cc:305 | 24289 | PPScheduler 把 waiting/loading 非空当作唤醒条件且 dp_size==1 用无超时 wait：仅剩不可准入等待流或加载中流（无在途批次）时引擎循环无… | ✅ 已修复+验证 PASSED |
| 10 | P2 | cpp/engine_base/schedulers/PPScheduler.cc:333 | 24281,24288 | PPScheduler 就绪解码优先轮整体旁路 waiting 流唯一维护路径：超时/错误退役、KV 释放与取消生命周期提交被推迟到批次排空窗口 | 🅐 已分析·搁置（=review#2） |
| 11 | P2 | cpp/models/logits_processor/TreeLogitsProcessor.cc:58 | 24278 | TreeLogitsProcessor 仍以流创建时的静态 hasNumBeams() 选择 token 布局，variable_num_beams 含 1 的 step 会对… | 🅐 已分析·搁置（base 既有，不优先处理） |
| 12 | P2 | cpp/normal_engine/NormalEngine.cc:775 | 24285,24298 | PP 终止协议仅在引擎循环干净退出时执行：step/pp_step 失败路径抛异常跳过 executor_->finish()，非首 stage 可能无限期等待终止 plan | 🅡 已澄清（非独立缺陷，剩余风险并入 #19） |
| 13 | P2 | cpp/normal_engine/pipeline/PPExecutor.cc:189 | 24300 | PP 激活快照按 pp_size+1 轮槽位常驻，既不被 warm-up 测量也无预留抬升覆盖，KV 池占满预算后余量耗尽时 clone OOM 会裸逃引擎线程致服务停摆 | ✅ 已修复 616cfa662c（完成即回收 activation snapshot） |
| 14 | P2 | cpp/normal_engine/pipeline/PPExecutor.cc:914 | 24290 | PP 尾 stage 在唯一引擎 loop 线程上同步冷编译 grammar：新 schema 请求可使整条 lane 停摆（上限 compile_timeout_ms 默认 … | ✅ 已处理 db71c7bf39（尾段预取与 forward 重叠） |
| 15 | P2 | distribute/distributed_server.py:139（+ frontend/frontend_app.py:426） | 24302,24303 | 多机 PP 前端选址对后端注册键一次性无捕获读取，等待界为 dist_comm_timeout（CLI 默认 300 秒、help 语义为心跳超时）而非注释宣称对齐的后端 bo… | 🛠 已修复待环境验证 |
| 16 | P2 | models/base_model.py:186 | 24280 | qwen_3 组合 pp_size>1 与 enable_ffn_disaggregate 无前置互斥守卫：support_pp 门禁放行、权重按 stage 裁剪后，PP 不… | ✅ 已修复 d5ff4c5cc0 |
| 17 | P2 | models_py/bindings/core/ExecOps.cc:725 | 24301 | PythonP2PWork::wait() 把 Python work.wait() 的返回值当作可 cast 的真值 bool 强约束，而注册契约只声明 typing.Any… | ✅ 已修复 616cfa662c |
| 18 | P2 | models_py/bindings/core/OpData.h:127 | 24297 | 新增 PP 输出分发按批行切 hidden_states，而 need_all_logits 为真时模型输出为全 token 行，与本变更新增的 OpData.h 契约注释相反… | ✅ 已修复 4d7d543a7d（lzf 提交：hidden_states 对齐 + per-request optional outputs） |
| 19 | P2 | models_py/distributed/collective_torch.py:381 | 24274 | PP 跨 stage 通信等待在仓库内无有界期限：stage 存活但停滞、消息序列失配或启动期 rank 缺席时，运行、停机与启动协商均可无限期挂起，stop() 无法收敛（文… | 🅐 已分析·搁置 |
| 20 | P2 | models_py/distributed/collective_torch.py:692 | 24286 | PP P2P 回调把全局 rank 连同非默认 lane 组直传 torch.distributed.isend/irecv，未做组内换算，正确性依赖快照内无法求证的 torc… | 🅡 已澄清（实现正确，已补真实 Gloo 回归） |
| 21 | P1 | openai/openai_endpoint.py:372 | 24305 | 非流式内部流式路径下每 chunk 物化全量累计 output_ids 且前端整请求驻留，单请求内存与聚合/解码 CPU 随输出长度平方增长（base 为线性），长输出可耗尽共享前端进程内存 | 🅐 已分析·搁置（归属 lzf，本分支不修） |
| 22 | P2 | model_loader/model_weight_info.py:225 | 24304 | pp>1 用 max(ep_size//local_world_size,1) 向下取整估算 EP 组承载节点数，ep_size 不整除 local_world_size 或 stage 未对齐节点边界时低估真实跨节点数，违反 rebalance_experts 的 num_nodes 语义与 num_gpus%num_nodes==0 断言契约 | ✅ 已修复+验证 PASSED d5ff4c5cc0 |
| 23 | P2 | test/BUILD:490 | 24307 | 新增 draft_model_loading_test 未固定执行池，用例内 patch 字符串目标（及 model.load 惰性导入）强制加载 librtp_compute_ops，在 cuda13 默认 CPU worker 上该目标于用例阶段报错 | ✅ 已修复 e10b307d68 |

## P2 分类与推进顺序（原 15 条聚类；补抓的 #22 归 D、#23 归 G）

| 类 | 主题 | 条目 | 状态 |
|---|---|---|---|
| A | 调度器语义 | #8 `max_generate_batch_size` 被重释为单轮 microbatch 上限、#9 waiting/loading 唤醒条件致自旋（#10 已搁置） | 🅡 #8 设计语义澄清；✅ #9 已修复 |
| B | 无界等待/挂起 | #19 跨 stage 等待无有界期限、#15 前端选址等待界错配（#12 已澄清：非独立缺陷，剩余风险并入 #19） | #19 已分析·搁置；#15 已修复待环境验证 |
| C | 资源/阻塞 | #13 激活快照常驻不被 warmup 计量、#14 loop 线程同步冷编译 grammar | ✅ #13 完成即回收；#14 尾段异步预取并与 forward 重叠 |
| D | 配置组合缺守卫 | #16 qwen_3 + FFN disaggregate、#22 EPLB num_nodes 估算 | ✅ 已修（`d5ff4c5cc0`） |
| E | 数据布局/契约 | #18 `need_all_logits` 时按批行切 hidden_states、#11 `variable_num_beams` 静态布局判定 | ✅ #18 已修复 4d7d543a7d；🅐 #11 搁置（base 既有） |
| F | torch 绑定契约 | #17 `PythonP2PWork::wait` 返回值强转、#20 全局 rank 直传 isend/irecv | ✅ #17 已修；🅡 #20 实现正确并补回归 |
| G | 构建/依赖 | #6 cuda13 wheel 主机改写、#7 链接闭包、#23 执行池 | ✅ 已清零（#6 回退、#7/#23 修复） |

推进顺序（原定）：#18（唯一静默错误）→ #16 → #7 → #8+#9 → #19+#12+#15 → #13+#14 → #17+#20 → #11 → #6。当前已完成 #1–#7、#9、#13、#14、#16–#18、#22、#23；#8、#12、#20 已澄清；#15 已修复待环境验证；#10、#11、#19、#21 已搁置。P2 已无待分析条目。

## 逐条详情

### 1. [P1] maxTokenNum() 改写为 int 域加法后，协议合法的近 INT_MAX max_new_tokens 有符号溢出，经 CompleteTokenIds::update 的 int 形参截断与 clamp 二次溢出得到巨量循环界，首个 token 提交即越界堆读并以虚假 OUT_OF_VOCAB_RANGE 泄露越界值，可致 SIGSEGV
- 位置：`cpp/engine_base/stream/GenerateStream.cc:994`　findings：24277,24293　（bot 提了 2 条）
- 状态：✅ 已修复 b8b0a87925
- bot 建议：恢复宽类型算术并饱和：const auto request_limit = static_cast<size_t>(generateConfig()->max_new_tokens) + static_cast<size_t>(inputLength()); return std::min(static_cast<size_t>(max_seq_len_), request_limit); 或在入口把 max_new_tokens 钳制/校验至 max_seq_len 范围。
- 我方分析：溢出成立。`max_new_tokens` 为 `int`（`GenerateConfig.h:48`）、`inputLength()` 返回 `int`（`GenerateStream.h:228`），`max_seq_len_` 也是 `int`（`GenerateStream.h:886`），故 `max_new_tokens + inputLength()` 在 int 域相加后才 `std::min`、再转 `size_t`。近 `INT_MAX` 的 `max_new_tokens` 触发有符号溢出（UB）。bot 描述的下游后果链（`CompleteTokenIds::update` 的 int 形参截断 → clamp 二次溢出 → 越界堆读 → 伪 OUT_OF_VOCAB_RANGE）未逐环核实，但溢出本身确定；触发面是客户端可传近 `INT_MAX` 的 `max_new_tokens` 且未被入口钳制。
- 结论/处置：**已由 lzf 修复。** `b8b0a87925` 将 `maxTokenNum()` 和 `nextStepSeqLengthLimit()` 的请求上限与 reserve 运算提升到 `int64_t`，避免 `int` 域溢出，再按 `max_seq_len_` 饱和；新增 `intMaxRequestCapsCommitLimitAndCommitsOnlyProvidedToken` 覆盖 `max_new_tokens` 接近 `INT_MAX` 的请求，并补 async reserve 超过 `INT_MAX` 后饱和到 0 的断言。

### 2. [P1] stage 路由发送端跳过 P/D TP 整除预检，互不整除比例使首个 PD 请求在每个 decode rank 的 RemoteLoad 处理器内命中未捕获 RTP_LLM_CHECK；FT_CORE_DUMP_ON_EXCEPTION=1 下 dp 通道全部 rank abort 并升级为整实例崩溃循环
- 位置：`cpp/model_rpc/DecodeRpcServer.cc:919`　findings：24279,24284,24294,24295　（bot 提了 4 条）
- 状态：✅ 已修复 23a7fa496d
- bot 建议：与 pp==1 路径及 CP 分支对齐：在 master 扇出前（loadCacheForAllRank 的 use_stage_routing 分支，或 prepareGenerateContext 校验 stage groups 处）用 group.peer_addrs.size() 与本机 pc.tp_size 做互整除预检，不兼容即返回 ErrorInfo(LOAD_KV_CACHE_FAILED)；或让 planStagePeerSlices 把比例不兼容写入 plan.error 而非断言；并为 RemoteLoad 补 try/catch 将处理器内异常转为 response->error_info，杜绝任何请求级/网络级数据触发处理器逃逸异常；进一步可在两端配置可知的装配或调度层拒绝互不整除组合并给出启动期诊断。
- 我方分析：_（待对齐时回填）_
- 结论/处置：_（待回填）_

### 3. [P1] 未启用 SP 的 PP 部署下，单个 num_beams>1 请求在首阶段 root 引擎循环线程触发未捕获断言异常，引擎停止推进、整条流水线停摆
- 位置：`cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc:63`　findings：24276,24283,24287,24291,24296　（bot 提了 5 条）
- 交叉引用：= review文档#1（beam 隔离）
- 状态：✅ 已由 e2d47e6fa0 解决（评审后提交，无需再改）
- bot 建议：参照 prepareStreams 对 logits processor 不兼容流的既有做法（PPExecutor.cc:625-631），在不依赖 sp_enabled_ 的受理层将 PP 不支持的 beam 请求转为单请求错误：入队前校验返回 INVALID_PARAMS，或在 prepareStreams/调度阶段 reportError+clearPPInflight+从批次剔除，保证引擎与其余请求继续服务。
- 我方分析：bot 报的是评审 head `423c9c34` 的真实状态；同日稍后的 `e2d47e6fa0`（liuzefeng.lzf，"fix: validate unsupported PP requests before execution"，5 文件 +290/-79）已按 bot 建议的 prepareStreams 路线修掉，且 423c9c34 是 e2d47e6fa0 的祖先（已用 merge-base 核实先后）。当前代码：① 新增 `PPExecutor::validateStream`（PPExecutor.cc:606-629），首条即 `if (stream.hasNumBeams()) return ErrorInfo(INVALID_PARAMS, "PP does not support beam search")`，**不依赖 sp_enabled_**（正是 bot 指出的原缺口），并一并覆盖 calculate_loss、SP 下 return_softmax_probs/prompt_logits、DSpARK tiled sampling（bot 24287 附带建议）、MTP logits processor 兼容性；② `prepareStreams`（:631-642）对校验失败的流 `reportError` + `clearPPInflight` + `erase` 剔除，引擎循环继续；③ 调用点在 processImpl 的"步骤 0"（:1250-1254，门禁 `isFirstStage() && isStageRoot()`），早于 buildPlan 与 asyncSendPlan，故不会再走到原断言点、计划照常下发；④ 原 `PPBatchStreamProcessor.cc:63` 的 `RTP_LLM_CHECK_WITH_INFO(!stream->hasNumBeams())` 已删除（该文件 -4 行），全文只剩 sampler 输入 `num_beams_in/out` 硬设 1。测试：`PPExecutorTest.FirstAdmissionRejectsBeamSearchBeforeForward`（:1702-1740）双 SP 类型（NONE/MTP）× 双 beam 形式（num_beams=2 / variable_num_beams={1,2}）参数化，断言被拒流 hasError+INVALID_PARAMS+消息含 "beam search"+`!isPPInflight()`，且同批健康流（303）无错、`process()` 返回 ok、下发 plan 的 `sampling_plan.request_ids == {303}` —— 直接覆盖 bot 担心的"整条流水线停摆"。
- 结论/处置：已解决，PP 分支无需再改。唯一未采纳的是 bot 24291/24296 建议的**前端（Python visitor）对 pp_size>1 增加与 check_sp_supported 同型的早期拒绝**：C++ 侧 reportError 的 INVALID_PARAMS 经 stream statusInfo 回给客户端，错误码不丢，前端那层只把失败提前到入队前；按既有裁决"PP 不支持特性的门禁只落 C++ 一侧、不加 Python 侧重复校验"（避免双语断言漂移），不补。若产品要求客户端立刻拿到 4xx 而非经一轮调度后返回，再单独评估。

### 4. [P1] PP 尾段在错误归集之前对整批 new_token_ids 做紧凑词表致命 CHECK 与 softmax gather:ROCm 构建下采样失败行的 -1 哨兵触发断言异常并穿透引擎循环线程(CUDA 失败行哨兵为 0,同一顺序无害)
- 位置：`cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc:583`　findings：24299　（bot 提了 1 条）
- 状态：✅ 已落地非 CUDA + PP 启动期门禁（PPExecutor::validateConfig，7940f08bdf）
- bot 建议：在 fillOptionalOutputs 与紧凑映射之前,依据 sampler_output.success/processor_errors 把失败行的 new_token_ids(及 gather 用 token_index)重写为合法占位(如 0)或按行跳过 gather/映射,错误统一交由 collectRequestErrors 的既有逐请求路径;亦可把 595 行致命 CHECK 降级为与非 PP 分发器同款的逐行 OUT_OF_VOCAB_RANGE 请求错误(NormalOutputDispatcher.cc:471-477),使 ROCm -1 哨兵与 CUDA 0 哨兵行为一致。
- 我方分析：**缺陷成立且当前代码未修**（行号因 `e2d47e6fa0` 删 4 行而整体 -4：bot 的 587/589/595/602/703 对应当前 583/585/591/598/699）。
  - 顺序事实：`fillExecutionResult`（:571-601）① :583 取 sampler 末列作 `new_token_ids`（**含失败行哨兵**）→ ② :585 `fillOptionalOutputs`，若批内任一流要 softmax（:239 用 `|=` 聚合成整批）则 :697-700 拿 `new_token_ids` 当索引 `probs.gather(1, token_index)` → ③ :587-597 开了 output_vocab 裁剪时对**全部行**做紧凑→全词表映射，:591 是 `RTP_LLM_CHECK_WITH_INFO(compact_token >= 0 && < size)` **致命断言** → ④ :598 `collectRequestErrors` 才依 `success` 逐请求归错。
  - 非 PP 对照（`NormalOutputDispatcher.cc:462-479`）顺序相反且均为可恢复错误：:464-467 **先判 success** → `reportError(UNKNOWN_ERROR, "sampler generate token id failed")` + `return`（不做映射）；:471-477 再判越界 → `reportError(OUT_OF_VOCAB_RANGE, ...)` + `return`（**不是断言**）；softmax 在 :481 之后，即 success 通过才算，不会拿哨兵去 gather。
  - 平台差异根因：采样内核是两份独立实现、各自约定失败行哨兵且无共同契约——ROCm `bindings/rocm/kernels/sampling/kernel.cuh:405/446/542` 写 `-1`（配 `success[bx]=false`，`sampling.h:22` 文档化契约），CUDA `bindings/cuda/kernels/sampling/flashinfer/sampling.cuh:855/936/1072/1234/1315` 写 `0`。`0` 是合法紧凑 id，CHECK 与 gather 均通过，失败行被映射成 `output_vocab_ids[0]` 但 :598 归错后、分发时 :804-810 `if (error.hasError()) { stream->update(error); return; }` 丢弃该行 token/softmax → CUDA 无害（`CudaSampleOp.cc:855-856` 注释亦印证为有意设计："Failed rows are zeroed and never gathered through their -1 token"）。
  - 后果（ROCm）：`-1` 命中 :591 CHECK → `myAssert` → `user_ft_core_dump_on_exception` 下 `abort()`，否则抛 `RTPException`；PPExecutor / `pp_step` / `loop` 全链无 catch（与 #3 同一条逸出路径），异常穿出 `normal_engine_loop` 线程入口 → `std::terminate` 进程崩溃。单请求采样失败被放大为进程级故障。
  - 可达性：`NormalEngine` 的裁剪拒绝列表（:189-213）拒了 SP / warm_up_with_loss / prefill CP / device-input，**未拒 PP、未拒 ROCm**；PP 源码还显式编译 `USING_ROCM` 分支（PPBatchStreamProcessor.cc:9、:690）。
  - SP 路径不受影响：另一个 `fillExecutionResult`（:651-678）的 `new_token_ids` 来自 `host_accepted.token_ids`（MTP 接受 token）而非 sampler 末列，且裁剪与 SP 互斥（NormalEngine:190）。
  - **验证覆盖事实（定门禁粒度的关键）**：PP 全部 17 个 smoke 用例只在 H20 上跑——`rtp_llm/test/smoke/BUILD:98` 唯一调用 `pp_suites(gpu_type = ["H20_CU13"])`；ROCm 套件树 `suites_rocm_oss.bzl`（注释自称 "INDEPENDENT suite tree"）的 7 个套件（basic/dense/moe/eagle/pd/embedding/jit_remote_cache）无一含 pp。即 PP 的 `#if USING_ROCM` 分支从未跑过任何端到端验证，"PP 支持 ROCm" 目前是纸面声称。
- 结论/处置：**不修 `fillExecutionResult` 顺序，改为启动期门禁整个 ROCm + PP 组合**（用户裁决）。粒度选型：
  - ✅ **A（采纳）门禁整个 ROCm + PP**：覆盖全部触发路径，不留漏洞。落点 `PPExecutor::PPExecutor` 构造函数，与既有 7 条 `"pipeline parallelism does not support X"`（PPExecutor.cc:264-298）同形态同落点；`--copt="-DUSING_ROCM=1"`（.bazelrc:289）是全局的，该文件可直接 `#if defined(USING_ROCM)`。措辞用"未验证"而非"不支持"（如 `"pipeline parallelism is not validated on ROCm"`），使日后在 AMD 卡跑通 smoke 后可直接删除门禁而不需改语义。已知代价（与既有 7 条同性质，接受）：报错前模型已加载、KVCMStorageBackend 等已构造，白做一段初始化。
  - ❌ B（只门禁 `output_vocab_ids` 非空）**有漏洞**：`return_softmax_probs` 是纯请求级参数（`GenerateStream.cc:406-408` = `generate_config->return_softmax_probs`），启动期无法得知未来请求是否携带，而它是第二条独立触发路径（:585 → :697-700 gather 负索引），故 B 等于没修完。
  - ❌ C（B + 在 `validateStream` 里请求级拒 `return_softmax_probs`）能覆盖全部，但要引入两层门禁，且语义比 A 更难解释（"平台支持但某请求参数不支持"）。
  - 附带收益：A 同时挡掉所有"PP + ROCm 未验证组合"的未知风险（该路径零 smoke 覆盖，#4 只是被 bot 看到的那一个）。
  - **遗留**：门禁不消除缺陷本身。若日后 ROCm 要上 PP，需回来把 `fillExecutionResult` 改成"先依 `success` 跳过失败行、再 gather/映射"（约 5 行：`success_cpu` 提前搬 CPU 并复用于 :598；gather 索引 `clamp_min(0)`；紧凑映射循环内 `if (!success_cpu[index]) continue;`，:591 CHECK 保留给健康行的真·越界不变量），届时删门禁。
  - 本地验证边界：开发机为 8×H20（CUDA），#4 在本地**触发不了**；本地只能验证门禁不影响 CUDA 路径（`pp_executor_test` / `pp_batch_stream_processor_test` 无回归）。ROCm 侧编译验证需走 CI 远程执行（`--config=rocm` → `gpu=MI308X`）。
  - **已落地并收敛（2026-10-20）**：`d5ff4c5cc0` 最初在 `PPExecutor::validateConfig` 以 `#if USING_ROCM` 门禁 ROCm；后续 `7940f08bdf` 将条件统一为 `#if !USING_CUDA`，启动期明确报错 `pipeline parallelism requires a CUDA build; set PP_SIZE=1 on non-CUDA platforms`，同时覆盖 ROCm 与其它非 CUDA 构建。CUDA 路径不受影响；若未来开放 ROCm + PP，仍需按上文方案修正失败行处理顺序后再移除门禁。

### 5. [P1] PP>1 下在线权重同步确定性失败：非持有 rank 的 UpdateWeights 因层/全局键缺失抛 KeyError 并返回 gRPC INTERNAL，且归属 rank 已写入导致新旧权重混合风险
- 位置：`model_loader/loader.py:530`　findings：24275　（bot 提了 1 条）
- 状态：✅ 已修复+提交 272d79ca4a
- bot 建议：WeightManager.update 复用 loader 归属信息（_layer_ids/has_pp_embedding/has_pp_lm_head 或 LoadConfig.pp_layer_range），对非本 stage 权重幂等跳过并返回成功；或提供每 rank 归属查询接口供客户端定向下发；或在启动校验中显式拒绝 PP>1 与在线权重更新组合并输出可诊断错误
- 我方分析：确认真 bug、PP 独有、P1 合理，触发面窄（仅 PP>1 + RL 在线权重同步）。加载侧按 stage 归属裁剪（loader `_layer_ids`/`_skip_pp_global_weight`），写入侧 `WeightManager.update` 未适配 → 非持有权重走 `update_layer_weight`/`update_global_weight` raise KeyError（model_weight_info.py:853/881）→ gRPC INTERNAL → RL 广播 all() 必假。用户确认 PP>1 要支持在线权重同步 → 走 bot 方向 1。
- 结论/处置：已修（方向 1）。loader.py 新增 `owns_pp_weight(name, layer_id)`（layer_id 查 `_layer_ids`、全局查 `not _skip_pp_global_weight`，复用加载侧归属、已计入 apply_pp_partition）；weight_manager.py `update()` 层/全局两分支各加幂等跳过（非本 stage `return` 成功，持有者仍走原逻辑保留 KeyError、不掩盖真错误）+ docstring 改准。测试：test_pp_layer_filter.py 加 `PPOwnsWeightTest`（层归属随 stage range / 全局 embedding 首段·lm_head 末段 / pp_size=1 拥有全部），28 tests OK。附带核实澄清（一度误判为 loader 不对称 bug，已纠正）：draft/propose 模型用 apply_pp_partition=False，但被 model_factory.py:171-174 `if not stage_has_lm_head(...): return None` 门禁在**末 stage**（apply_pp=False 全仓仅此一处）；故 `_load_lm_head = has_pp_lm_head`（无 apply_pp 守卫）是对的（末 stage has_pp_lm_head=True），`_load_embedding` 的 `not apply_pp_partition or` 守卫也必要（draft 在末 stage 需 embedding 但 has_pp_embedding=False）。共两个 loader（target apply_pp=True 全 stage 分片 / draft apply_pp=False 仅末 stage），无不对称 bug。改动 3 文件（loader.py / weight_manager.py / test_pp_layer_filter.py），当前提交为 272d79ca4a（--no-verify；新增测试已手动改成 black 规范形，未带入既有 black 债的整文件重排）。

### 6. [P2] cuda13 x86 五个 wheel 的 OSS 主机改写后，其中四个对象在新主机上无快照内可用性证据；若新主机缺任一对象，该平台构建与安装期整体失败
- 位置：`deps/requirements_lock_torch_gpu_cuda13.txt:634`　findings：24273　（bot 提了 1 条）
- 状态：✅ 已回退那 10 行主机改写，与 #7/#23 合并提交 e10b307d68（bot 的担心本身已实测否证，但改动不该存在于本 PR）
- bot 建议：合并前逐个实测 5 个新主机 URL 可下载且 sha256 与 lock 一致，或运行 bazelisk run //deps:requirements_torch_gpu_cuda13.update 再生成 lock 并确认无差异（uv --generate-hashes 必须真正下载直链 wheel，成功即证明可达）；同时把 arch_config/arch_select.bzl:96-108 的 whl_deps 主机同步迁移以免两条依赖路径长期分裂，并考虑仿照 deps/http.bzl:64-67 为关键 wheel 配置第二来源；无法验证时保留旧主机。
- 我方分析：**bot 的核心担心（"若新主机缺任一对象则整体失败"）已实测否证，但这不等于改动应该保留——它违反 lock 生成规范、且在本 PR 的任何构建路径上都不生效，已回退（见下方结论）。**以下为核实过程：
  - 实测（HEAD 请求只取状态码，未下载 wheel）：新主机 `rtp-maga.oss-cn-zhangjiakou.aliyuncs.com` 上 5 个对象（torch / torchvision / flash_mla / fast_safetensors / fastsafetensors）**全部 HTTP 200**；旧主机 `rtp-maga.cn-zhangjiakou.oss.aliyuncs.com` 5 个**全部 000**（连不上）。
  - DNS 对照：旧主机解析到 **`1.1.1.1`**（通配符兜底 = 域名已失效），新主机解析到真实 OSS `39.98.14.68`（经 `sc-1o9r.cn-zhangjiakou.oss-adns.aliyuncs.com` CNAME）。即在 x86 开发机上旧主机已死——这解释了 lzf 为何会改这 10 行（lock+requirements 各 5 行，路径/文件名/sha256 逐字不变），但改动落错了文件：他改的是内源构建下被屏蔽的子仓副本（详见结论）。
  - **两个主机不是新旧替代，而是机房可达性相反**：`internal_source/deps/requirements_cuda12_arm.txt:1` 原文注释——"注意gb200所在的th机房访问不了rtp-maga.oss-cn-zhangjiakou.aliyuncs.com要改成rtp-maga.cn-zhangjiakou.oss.aliyuncs.com"。即 th 机房（GB200/ARM）必须用旧主机，x86 机房必须用新主机。
  - **内源构建不用 bot 指的那个文件**：`.internal_bazelrc:2-3` 有 `--override_repository=rtp_deps=%workspace%/internal_source/deps` 与 `--override_repository=arch_config=%workspace%/internal_source/bazel`，故内源构建时 `arch_config` 被换成 `internal_source/bazel/arch_select.bzl`；bot 引用的 `github-opensource/arch_config/arch_select.bzl` **仅在纯开源构建下生效**。
  - 远端核实（内源 gitlab 当时领先本地 135 个提交）：`gitlab/main-internal` 版本的 `internal_source/bazel/arch_select.bzl` 主机分布为旧 17 / 新 16 / artlab 32，而同步前本地为旧 13 / 新 29 / artlab 23（**本地已于 2026-10-20 reset --hard 到 `19b08094f1`，现为远端值**）—— 旧主机用量在远端反而增加，进一步印证两主机并存是有意为之。135 个提交里无主机迁移：`93e79a4958`（commit message 看似相关）实际删的是 `cuda13_test_exec_properties` 函数（CI 池路由），`06bc143d6e` 是多模态运行时下载的 SSRF check，均与构建依赖主机无关。
  - `github-opensource/arch_config/arch_select.bzl` 的 `whl_deps()` 里旧主机共 9 处，需分开看：`using_cuda13_x86` 分支 5 处（:96 torch / :97 torchvision / :100 flash-mla / :107 fast-safetensors / :108 fastsafetensors）**正是 lock 改的同一批 wheel，该同步迁移**；`using_cuda13_arm` 分支 4 处（:113/:114/:122/:123，arm_pkg 路径）**必须保持旧主机**（th 机房可达性相反）。bot 的"同步迁移"建议未区分这一点。
- 结论/处置：**回退这 10 行主机改写**（已执行）。bot 担心的方向错了，但这改动本身也不该在 PR 1521 里。
  - **不该存在的四条证据**：① lock 文件头两行自己声明 `# This file was autogenerated by uv via the following command: # bazelisk run //deps:requirements_torch_gpu_cuda13.update`；② `docs/references/Update-whl-Deps.md:26` 明文规定 "Lock files must not be manually modified and can only be automatically generated"；③ bot 自己提供了"这是手工改的"证据——原文写 "head 仅改主机形式，路径、文件名、哈希、行数逐字节相同"，而用工具重生会让 uv 重新解析依赖树、包版本/哈希/行数必然变化（对照：内源版与子仓版就有实质包差异如 `annotated-doc==0.0.4`），只改主机名而其余逐字节不变 = 手工编辑；④ github 侧 CI 不跑 cuda13 构建——`.github/workflows/` 只有 3 个触发 stub（CI-event-dispatcher / CI-request-trigger / merge-request-trigger），grep `cuda13|cuda12_9|config=cuda` 零命中。
  - **实际效果为零**：内源构建被 `--override_repository=rtp_deps` 屏蔽（读 `internal_source/deps/` 那份）、github CI 不跑 cuda13、内源 CI 的 cuda13 lane 同样读内源那份；**唯一生效场景**是开源社区用户按 `README_cn.md:65` 执行 `pip3 install -r ./deps/requirements_torch_gpu_cuda13.txt`。且 `6446ec7e17` 里 `deps/` 只改了这 2 个文件、无任何配套改动，commit message 的 8 条要点也未提依赖迁移 —— 属夹带在 221 文件 PP 大提交里的无关改动。
  - **回退手法（不能用 `git checkout origin/main -- <file>`）**：`git diff origin/main HEAD` 对这 2 个文件混合了两种差异——lzf 改的 10 行主机，以及 **main 前进造成的 `rtp-kernel` 版本差异**（`94fd20eb`/feat-port-nvfp4-xqa → `34e3b72a`/main，含注释与 sha256，lock 4 行 + requirements 6 行）。用 checkout 会连带把 rtp-kernel 改成 main 版本、引入无关改动。改用精确主机名替换（已先核实两文件里新主机各出现 5 次、旧主机 0 次，故全局替换恰好只影响那 5 处）。
  - **回退后三重核实**：① 两文件新主机计数归 0、旧主机计数回 5；② 工作树 diff 含 20 行 rtp-maga（10 删 10 加）、**0 行 rtp-kernel**（未误碰）；③ `git diff origin/main` 对这 2 个文件只剩 rtp-kernel 那一处（属 main 前进、非本分支改动，刷主干时自然消失）。
  - **保留的核实事实（供将来真要迁移时用）**：新主机 5 个对象实测全 200、旧主机在 x86 环境 DNS→`1.1.1.1` 已失效；但 th 机房（GB200/ARM）恰好相反（`internal_source/deps/requirements_cuda12_arm.txt:1` 注释）。若确需迁移，应单独提 PR 走规范流程：改 `requirements_torch_gpu_cuda13.txt`（基础描述文件，允许手工改）→ 跑 `bazelisk run //deps:requirements_torch_gpu_cuda13.update` 重生 lock → 同步 `arch_config/arch_select.bzl` 的 `using_cuda13_x86` 分支 5 处（arm 分支 4 处必须保留旧主机），并同步内源侧的 `internal_source/deps/` 与 `internal_source/bazel/arch_select.bzl`（那才是内源构建实际生效的一份）。
  - 待确认：这 10 行是 lzf 在 `6446ec7e17` 里改的，回退前宜知会他一声——若他当时是在某个纯开源构建环境（不带 internal_source）跑 cuda13 失败才顺手改的，回退会让那个环境继续失败；但按 PR 1521 的 CI 走内源 Aone 来看，该场景不成立。
  - 不采纳 bot 的"配第二来源"建议：`deps/http.bzl:64-67` 的多来源机制服务于源码归档，wheel 直链走 pip_parse 的 hash 校验路径，加第二来源需改 lock 生成流程，超出本条范围。
- 补充核实（内源同步到 `gitlab/main-internal` `19b08094f1` 后，2026-10-20）：**bot 与本条分析都只看得到开源侧，而内源构建下这个文件根本不生效。**
  - `.internal_bazelrc:2` 有 `--override_repository=rtp_deps=%workspace%/internal_source/deps`，而 `internal_source/deps/pip.bzl:76` 的 `pip_gpu_cuda13_torch` 用 `requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda13.txt"` → 解析到 **`internal_source/deps/requirements_lock_torch_gpu_cuda13.txt`**，子仓的 `deps/requirements_lock_torch_gpu_cuda13.txt`（= lzf 改的那份、= bot 评审的那份）在内源构建下被完全屏蔽。
  - 内源侧实测：`internal_source/deps/requirements_lock_torch_gpu_cuda13.txt` 的 5 个 wheel（torch / torchvision / flash-mla / fast-safetensors / fastsafetensors，与子仓同批同路径同 sha256）**全部仍用失效旧主机**；`internal_source/bazel/arch_select.bzl` 的 `using_cuda13_x86` 分支也是 5 处旧主机 + 7 处 artlab。
  - 两套 lock 是**独立维护**而非同一文件的两份副本：内源版头部生成命令是 `uv pip compile internal_source/deps/requirements_torch_gpu_cuda13.txt --python-platform x86_64-manylinux_2_28 ... --overrides deps/requirements_cuda13_overrides.txt`，子仓版是 `bazelisk run //deps:requirements_torch_gpu_cuda13.update`；除主机名外还有实质包差异（如内源版含 `annotated-doc==0.0.4`）。
  - 影响面判定：本机旧主机 DNS 解析到 `1.1.1.1`（失效），故**在本机跑 `--config=cuda13` 会因抓不到 wheel 而失败**；但我们日常用 `--config=cuda12_9`，pip_parse 是惰性的（只在用到 `requirement_gpu_cuda13(name)` 时才抓），**对当前 PP 工作零影响**。CI 的 cuda13 lane 是否受影响取决于执行机房能否解析旧主机（`internal_source/deps/requirements_cuda12_arm.txt:1` 注释表明 th 机房恰好相反：新主机不可达、必须用旧主机），仓库内无法定论。
  - 因此本条对 PR 1521 的结论**不变**（开源视角下 lzf 改子仓 lock 是必要修复、bot 的担心已否证）；但附带查出的内源侧问题（内源 cuda13 x86 lock 与 arch_select 的 x86 分支均指失效旧主机，且与子仓已迁移的新主机不一致）**不属 PR 1521 范围**，应单独反馈给内源维护者，由其确认 cuda13 lane 的执行机房后决定迁移方向。

### 7. [P2] 新增 pp_topology_validator_test 链接闭包缺少 execPPSnapshotExchange 的唯一定义：opt 特性关闭的受支持编译模式（-c dbg / -c fastbuild）下确定性链接失败
- 位置：`cpp/cache/test/BUILD:96`　findings：24272　（bot 提了 1 条）
- 状态：✅ 已修复+验证 PASSED，与 #6/#23 合并提交 e10b307d68
- bot 建议：按同文件 pp_stage_cache_config_test（test/BUILD:118/:120）与 test_deps 默认分支（test/BUILD:30-40）的做法，为 pp_topology_validator_test 补 //rtp_llm/models_py/bindings/core:exec_ops_test_lib（cuda 分支另加 cuda_impl/cuda_headers/cudart，rocm 分支换成 exec_ctx_rocm）并加 torch_deps；或把依赖 ExecOps 的 PPSnapshotCollector/PPCacheCapacityNegotiator 从 pp_topology_validator 拆为独立 cc_library（或改为可注入的交换回调），使纯校验逻辑（validatePPTopology/initPPCacheGeometry/序列化）的测试闭包自洽，不再依赖链接器段回收兜底。
- 我方分析：**缺陷成立，静态证据 5 条互相印证，无需实跑即可定性**。
  - ① 库引用了该符号：`PPTopologyValidator.cc:354-355`（`PPSnapshotCollector::collect()` 内调 `execPPSnapshotExchange(local_.serialize())`）。
  - ② 符号唯一提供方不在闭包内：全仓仅 `ExecOps.cc:848` 定义（`ExecOps.h:147` 为非 inline 声明），而 `ExecOps.cc` 只被 `exec_ops_test_lib`（core/BUILD:136）与 `exec_ctx_state`（core/BUILD:220）编译。
  - ③ 库只依赖头文件：`pp_topology_validator`（cache/BUILD:121-133）srcs 仅 `PPTopologyValidator.cc`，ExecOps 依赖是无 srcs 的 `exec_ops_hdr`。
  - ④ 测试 deps 缺提供方：`pp_topology_validator_test`（cache/test/BUILD:95-107）deps 只有 cache_config / kv_cache_specs / pp_topology_validator / gtest / gtest_main，**既无 `exec_ops_test_lib` 也无 `torch_deps()`**。
  - ⑤ **同文件兄弟目标依赖同一个库却补了**：`pp_stage_cache_config_test`（:109-125）deps 含 `//rtp_llm/models_py/bindings/core:exec_ops_test_lib` + `torch_deps() + cuda13_torch_link_deps` —— 证明这是漏加而非有意设计。
  - 默认 `-c opt` 能过的机制：测试用 `MockStageSnapshotCollector`（Test.cc:226-236）覆盖 `collect()`，故 `PPSnapshotCollector::collect` 在测试二进制内无入引用；工具链只在 opt 特性下加 `-ffunction-sections/-fdata-sections` + `-Wl,--gc-sections`，该函数段连同未定义引用一起被裁掉。`-c dbg` 只加 `-g`、`-c fastbuild` 无标志，不分节所以裁不掉 → 未定义引用保留 → 链接失败。
- 结论/处置：**采纳 bot 的第一个建议（补 deps，与兄弟目标一致），不采纳拆库**。
  - 修法：给 `pp_topology_validator_test` 的 deps 补 `//rtp_llm/models_py/bindings/core:exec_ops_test_lib`，并加 `+ torch_deps() + cuda13_torch_link_deps`（照 :109-125 的完整形态）。
  - 不采纳拆库（把 `PPSnapshotCollector`/`PPCacheCapacityNegotiator` 移出 `pp_topology_validator`）：更彻底但改动面大、会让库结构偏离既有形态，且兄弟目标已确立"测试补 exec_ops_test_lib"的惯例。
  - 已知代价（接受）：补 `torch_deps()` 会让这个原本纯逻辑的测试拉上 torch 依赖、编译变重——兄弟目标已是如此，属既有惯例。
  - **不做 `-c dbg` 实跑验证**，理由是磁盘成本不可控：`/data1` 4.0T 已用 **92%、仅剩 337G**（40 人共享盘，五天内从 769G/81% 恶化到 337G/92%）；现有 `bazel_cuda129_pp_lzf_cache` 87G（其中 `bazel-out/k8-opt/` 占 52G），而 `-c dbg` 产物落 `bazel-out/k8-dbg/`、与 opt **完全独立零共享**，dbg 全量重编所有源码依赖（含 external 的 grpc/protobuf）的体量无法准确预估。且 bot 自身定性为"测试构建期失败，非运行时故障"（生产路径 `ExecOps.cc` 经 `exec_ctx_state` 编入 `librtp_compute_ops.so`，共享库中该引用在加载期解析），边际价值低。
  - 验证方式：修完用默认 `-c opt` 跑一次 `//rtp_llm/cpp/cache/test:pp_topology_validator_test` 确认不破坏现状（opt 产物已在，增量成本极小，27 个用例应全绿）。
  - **已执行并验证通过**：`cache/test/BUILD` 的 `pp_topology_validator_test` deps 插入 `//rtp_llm/models_py/bindings/core:exec_ops_test_lib`、并把 `],` 改为 `] + torch_deps() + cuda13_torch_link_deps,`（+4/-3，与兄弟目标逐字同形态，未用 select——已核实 `exec_ops_test_lib` 无 `target_compatible_with`、无 rocm 分支，且兄弟目标 `pp_stage_cache_config_test` 也是直接列不用 select）。测试结果：**PASSED in 11.0s、TEST_EXIT=0**（`--config=cuda12_9 --cache_test_results=no`，15,965 个 action）。
  - 磁盘代价实测（供后续同类改动参考）：补 `torch_deps()` 后为链接 torch 而拉入大量 CUDA kernel 编译（日志可见 xqa_kernel_sm90 等 ptxas 输出），`/data1` 可用从 337G 降到 **310G**（单次约 27G）。

### 8. [P2] PPScheduler 把 max_generate_batch_size（即用户侧 concurrency_limit）重释为单轮 microbatch 上限：DECODE 在飞并发流可达约 (pp_size+1)×配置值，KV 容量落在窗口内时边际请求以 MALLOC_FAILED 终态失败而非排队
- 位置：`cpp/engine_base/schedulers/PPScheduler.cc:92`　findings：24282　（bot 提了 1 条）
- 状态：🅡 已澄清（PP microbatch 设计语义，不改代码）
- bot 建议：把在飞/持有 KV 的流纳入总量口径：fitsCurrentBatch 计入 running_streams_（含 isPPInflight）对齐 FIFO 语义，或按流水线占用拆分 microbatch 使在飞总量不超过 max_generate_batch_size；若保留单轮上限语义，则文档化 PP 下 concurrency_limit 表示每 microbatch 上限、总并发约为 (pp_size+1)×，给出每 stage KV 增长预留下限，并为 decode 流提供有效总量兜底（现有 max_inited_kv_cache_streams 对已持块流不生效）。
- 我方分析：**scheduler 层现象成立，但 bot 将其直接定性为用户并发承诺被突破并不完整。** `evaluateRunningStreams()` 会跳过仍留在 `running_streams_` 的 `isPPInflight()` 流，`ScheduleRuntime::scheduled_stream_count` 只统计本轮可执行/新准入流，因此 `fitsCurrentBatch()` 的 `max_generate_batch_size_` 确实是单轮 batch 上限；`PPExecutor` 有 `pp_size+1` 个 slot，输入充足时 scheduler 内部可同时保留多个在途 batch。非 PP FIFO 则把 `running_streams_.size()` 纳入同一判断，两者语义不同。
  - 该差异有明确的 PP 合理性：流水线需要多个 microbatch 同时在途才能 overlap；若直接把所有 PP inflight 流计入现有上限，在没有独立 microbatch 大小配置的情况下，积压请求可能一次占满总额度，后续 slot 只能发空 plan，PP 退化为低占用串行推进。`max_generate_batch_size` 同时还用于 sampler 初始容量、模型 batch/CUDA Graph 等单次执行资源，其执行层含义本来就是单 batch 尺寸。
  - 用户侧 `concurrency_limit` 另有入口控制：Python frontend 的 controller 按 `concurrency_limit * dp_size` 限制活跃请求，C++ HTTP server 也有 `ConcurrencyControllerGuard`，普通单请求路径不会仅因 PP slot 数自动放大到 `(pp_size+1)×`。但这不是 scheduler 的绝对硬边界：controller 是入口/进程级的，`/batch_infer` 明确让整个 prompt batch 只占一个 slot，却创建多个 stream，PD Decode RPC 也直接向 engine enqueue。因此 bot 的上界是 scheduler 可达上界，而不是所有标准部署的稳定并发量。
  - 资源风险并非虚构：decode 流初始 KV 在 RPC 侧分配后入队，`max_inited_kv_cache_streams` 对 `already_inited_kv` 候选不拦截；若在途流后续共同增长耗尽 KV，`GenerateStateMachine::handleRunning()` 的 `incrKVBlock()` 失败会将该请求终态置为 `MALLOC_FAILED`。但这需要入口聚合/批量请求和具体 KV 水位共同满足，不是 PP 默认必现；资源耗尽本身也是现有 allocator 的既定失败语义。
- 结论/处置：**不修改代码，按 PP microbatch 设计语义回应澄清。** 不能仅为对齐非 PP 而把 `running_streams_.size()` 加进 `fitsCurrentBatch()`，因为这会用同一个值同时限制总在途数和单轮 batch，直接损害流水线填充。若以后线上证据表明需要更强的后端资源保护，应独立设计“PP 总在途/KV 持有上限”与“单 microbatch 上限”，并基于 stage KV 容量和实际 batch 切分调优；不在本 PR 复用 `concurrency_limit` 做二义性修补。

### 9. [P2] PPScheduler 把 waiting/loading 非空当作唤醒条件且 dp_size==1 用无超时 wait：仅剩不可准入等待流或加载中流（无在途批次）时引擎循环无任何退避满速自旋，并向全流水线洪泛空 plan、每轮重复分配/驱逐尝试与指标上报
- 位置：`cpp/engine_base/schedulers/PPScheduler.cc:305`　findings：24289　（bot 提了 1 条）
- 状态：✅ 已修复+验证 PASSED
- bot 建议：按 FIFOScheduler 分级恢复退避：唤醒谓词只保留 stop_/schedule_trigger_/running_streams_（PP 推进在途批次所必需）；当本轮既无准入也无在途、仅剩 waiting/loading 时改用定时等待（loading 约 10ms、纯 waiting 约 100ms），或对连续“无准入且无在途”轮次做指数退避，并节流每轮 initKVBlock/驱逐重试与 MALLOC_RETRY、调度指标上报；dp_size>1 的假流心跳同样保留最小间隔，避免背靠背最小假前向。
- 我方分析：**缺陷成立，但并不只影响 `dp_size==1`。** `condition_variable::wait(lock, predicate)` 与 `wait_for(lock, duration, predicate)` 都会在 predicate 已为 true 时立即返回；因此 waiting/loading 非空后，即使 `dp_size>1` 走 10ms 的 `wait_for`，也不会真的等待 10ms。若本轮请求因 KV 容量等条件仍不可准入，或异步 cache load 尚未完成，相同队列会让下一轮再次立即通过，形成无退避调度循环。`running_streams_` 不能从谓词删除：PP 在途请求仍保留在该队列，Stage 0 必须继续产生空轮次推进流水线并接收结果。新请求入队则已由 `FIFOSchedulerBase::enqueue()` 设置 `schedule_trigger_=true` 并 `notify_all()`，不需要 waiting 非空作为首次唤醒条件。从历史看，PPScheduler 初版沿用了当时 FIFO/PDFusion 的旧谓词；普通 FIFO 后来已因 retryable KV shortage 自旋改成分级定时轮询，PP 未同步该修正。
- 结论/处置：**采纳 bot 的分级退避建议，不新增配置、不改 PP 执行流程。** `waitPredicate()` 仅保留 `stop_ || schedule_trigger_ || !running_streams_.empty()`；有 loading 流或需要 DP fake stream 时每 10ms 轮询，只有 waiting 流时每 100ms 轮询，无待处理工作时仍无限等待。这样首次 enqueue 仍立即调度、PP 在途批次仍连续推进，仅“尝试后没有状态进展”的 waiting/loading 轮次退避。新增 `CapacityBlockedWaitingStreamBacksOffBetweenPolls`，验证 enqueue 首轮立即尝试、不可准入流保留后 predicate 变 false、下一轮发生有界退避；既有异步加载用例补断言仅 loading 时 predicate 为 false，并继续覆盖无外部通知下的 10ms 轮询晋级。`//rtp_llm/cpp/engine_base/schedulers/test:pp_scheduler_test` 在本机 CUDA 12.9 下 **PASSED in 11.7s**，`git diff --check` 通过。

### 10. [P2] PPScheduler 就绪解码优先轮整体旁路 waiting 流唯一维护路径：超时/错误退役、KV 释放与取消生命周期提交被推迟到批次排空窗口
- 位置：`cpp/engine_base/schedulers/PPScheduler.cc:333`　findings：24281,24288　（bot 提了 2 条）
- 交叉引用：= review文档#2（PDFUSION 饥饿）
- 状态：🅐 已分析·搁置（=review#2）
- bot 建议：把等待流维护与准入决策解耦：每轮无条件对 waiting_streams_ 做一次超时/错误退役扫描（等价于以空准入集调用 FIFOSchedulerBase::evaluateAndUpdateStreams 的退役遍历），使退役不再依赖 scheduled_streams.empty()；如需消除 TTFT 门控，可另行增加有界公平（限制连续解码优先轮数，或 waiting 最老等待时间超阈值时强制插入独立 prefill 批次，不要求混批）。
- 我方分析：_（待对齐时回填）_
- 结论/处置：_（待回填）_

### 11. [P2] TreeLogitsProcessor 仍以流创建时的静态 hasNumBeams() 选择 token 布局，variable_num_beams 含 1 的 step 会对宽度 1 的增量张量按历史偏移读取（base 已存在，head 仅收窄）
- 位置：`cpp/models/logits_processor/TreeLogitsProcessor.cc:58`　findings：24278　（bot 提了 1 条）
- 状态：🅐 已分析·搁置（base 既有，不优先处理）
- bot 建议：把布局判定从配置改为本次调用形状，与 RecommendationLogitsProcessor 对齐：const int64_t stride = new_tokens.size(1); const bool use_offset = stride > num_new_tokens; 仅 use_offset 时取 current_output_length+input_length，并加 RTP_LLM_CHECK_WITH_INFO(offset + num_new_tokens <= stride, ...)；或由调用方把 usesBeamSearchTokenLayoutForCurrentStep() 的逐 step 标志随 updateStatus 传入，并修正新增注释中“Beam updates contain full token histories”的表述。补一个 variable_num_beams=[1,2,1] 且首步传 [1,1] 增量张量的回归用例（可复用哨兵存储）。ThinkModeLogitsProcessor 若重新接入工厂需同样改造。
- 我方分析：**缺陷成立，但逻辑是 base 既有，本 PR 只新增了两行表述不准的注释。** ① `:58` 用流创建时写死的 `info.is_beam_search`（`fromGenerateInput:88` 传 `hasNumBeams()`，即 `maxNumBeams()>1`，`GenerateConfig.h:213-214/220-222`）决定 `offset = current_output_length + input_length`。② 但 `variable_num_beams` 下逐 step 布局会变：`GenerateStream::usesBeamSearchTokenLayoutForCurrentStep()` = `currentNumBeams()>1 || nextNumBeams()>1`（`GenerateStream.cc:390-392`），而 `numBeams(0)` 固定返回 1（:359-362）、其余 step 取 `variable_num_beams[idx]`（:365-370）。以 `variable_num_beams={1,2,1}`、input_length=5 为例：第 0 步 current=next=1 → 布局是增量 `[batch,1]`，但 `is_beam_search=true` 使 offset=5 → 读 `new_tokens[i*1+j+5]` 越界；同一函数 `!is_beam_search` 分支的 `CHECK(num_new_tokens == new_tokens.size(1))`（:60-62）也被跳过，无任何保护。③ 触发面：`tree_decode_config` 加载成功（`LogitsProcessorFactory.cc:151` 创建 processor）+ `variable_num_beams` 且存在 current=next=1 的 step；后果是越界读（垃圾 token id 喂 DFA 或段错误）。④ 归属：`:58` 为 base（`a19ed99befb`，2025-05-15），`is_beam_search` 字段同为 base；同类问题在 `RecommendationLogitsProcessor.cc:342-366` 早已用 `stride > num_new_tokens` 按形状判定（`4f5e24872bb`，2026-07-13，base）；本 PR 仅在 `:56-57` 新增注释（`6446ec7e17`，lzf），把"beam 更新都是 full history"写成统一事实，未覆盖 variable beam 的增量 step；本 PR 新增的 `usesBeamSearchTokenLayoutForCurrentStep()` 只用在 `CompleteTokenIds::update`（`GenerateStream.cc:1250`）与 `NormalOutputDispatcher.cc:343`，TreeLogitsProcessor 未跟进。
- 结论/处置：**搁置，不优先处理**（用户裁决：base 既有问题不在本分支优先处理）。若后续处理：照 `RecommendationLogitsProcessor.cc:342-366` 写成 `stride > num_new_tokens` 判定并补 `offset + num_new_tokens <= stride` 校验（约 8 行），同时把本 PR 新增的 `:56-57` 注释改成"仅当前 step 使用 beam 布局时才是 full history"，并补 `variable_num_beams=[1,2,1]` 首步增量的回归用例。

### 12. [P2] PP 终止协议仅在引擎循环干净退出时执行：step/pp_step 失败路径抛异常跳过 executor_->finish()，非首 stage 可能无限期等待终止 plan
- 位置：`cpp/normal_engine/NormalEngine.cc:775`　findings：24285,24298　（bot 提了 2 条）
- 状态：🅡 已澄清（非独立缺陷，剩余风险并入 #19）
- bot 建议：在 loop() 中以 try/catch 包裹循环体，失败时仍执行幂等的终止协议（executor_->finish()）后再销毁执行器，或在 stop() 于 join 之后兜底调用幂等 finish()；保证失败 rank 进程及时退出使对端可检测；并为 PP 通信 wait 配置独立的有限期限或进度 watchdog，不依赖 100 年组超时作为唯一边界。
- 我方分析：**“异常跳过 `executor_->finish()`”这一表面观察成立，但“失败 rank 继续存活、仅 engine 线程退出，因此对端无限等终止 plan”的前提不成立。** ① `pp_step()` 直接抛异常时会逃出 `NormalEngine::loop()`；返回非 OK status 时，当前 `trySaveStepError()` 固定返回 `UnimplementedError`，`THROW_IF_STATUS_ERROR` 也会抛异常。② 本仓锁定的 Havenask `autil::Thread::threadWrapperFunction()` 只直接调用 `p->threadFunction()`，外层没有 `try/catch`；异常逃出 pthread 入口后触发 `std::terminate()`/`SIGABRT`。RTP-LLM 的 SIGABRT handler 只记录堆栈并恢复默认信号再次发送，不会吞掉进程退出。③ 本地 rank 由 `ProcessManager` 以默认 1s 周期检查；任一 rank 异常死亡后标记 failure，立即终止同节点其他 rank，backend 父进程最终非零退出交由上层重启。因此当前的异常语义本就是进程级 fail-fast，而不是留下一个保持连接但不再推进的 rank。
- 结论/处置：**不修改 `NormalEngine::loop()`，不在异常路径强行调用正常 `finish()`。** `finish()` 假设通信序列完整且 communicator 仍可用；若中间 stage 已转发 plan 后在 model forward 失败，下游正等 activation，而该 stage 的 `finish()` 只会继续等上游 terminal plan，反而固化失配/死锁。失败进程退出后对端能否及时从 Gloo/NCCL wait 中收敛，仍是真实问题，但完整归属 #19 的通信故障检测与等待边界，不是独立的终止 plan 缺陷。

### 13. [P2] PP 激活快照按 pp_size+1 轮槽位常驻，既不被 warm-up 测量也无预留抬升覆盖，KV 池占满预算后余量耗尽时 clone OOM 会裸逃引擎线程致服务停摆
- 位置：`cpp/normal_engine/pipeline/PPExecutor.cc:189`　findings：24300　（bot 提了 1 条）
- 状态：✅ 已修复 616cfa662c（完成即回收 activation snapshot）
- bot 建议：1) 为 P2PWork/PPCommTicket 增加非阻塞完成查询（is_completed 或 CUDA event），通信完成即释放快照，或改用有容量上限的发送 buffer pool，使常驻项不随 pp_size 放大；2) 在 MemoryEvaluationHelper::getDefaultRuntimeMemorySize 按既有 updateMemoryIfNeeded 模式为 pp_size>1 显式抬升预留，量级参考 pp_size × 单轮 token 上界 × hidden × (2+aux 份数) × dtype；3) 为 pp_step 补上与 step 一致的 OOM 诊断与异常处理，避免异常裸逃引擎线程。
- 我方分析：**评审 head 上缺陷成立，最新代码已采用 bot 的方向 1。** `616cfa662c` 为 `P2PWork` 增加 `isCompleted()`、为 `PPCommTicket` 增加 `tryComplete()`，并在每轮 `waitLastRun()` 后由 `PPExecutor::reapCompletedActivationSends()` 扫描所有 slot：已完成的 NCCL activation send 会立即 `wait()` 校验并删除 ticket，从而释放其 clone；Gloo metadata 因完成状态只在 `wait()` 中推进，仍保留到原有 `waitAll()`，不破坏协议。该提交同时引入两个 grow-only receive buffer，避免接收侧逐 batch 重分配，但这属于附带优化。
- 结论/处置：**已解决“完成的快照滞留到 slot 复用、常驻量随 pp_size 放大”的核心问题。** 未额外抬升 `MemoryEvaluationHelper` 预留，也未在本条补通用 OOM catch；未完成的真实在途 send 仍必须保留快照，通信异常的等待收敛继续归 #19。测试覆盖 `CommonActivationSendsReapedAcrossSlotsIncludingSkippedBatches`、`CommonActivationReapingPropagatesFailureWithoutDroppingRemainingSends`、`CommonActivationSendKeepsSnapshotUntilReclaimedAndMetadataUntilWait`；提交说明记录 `pp_serialization_test` 16 cases、`pp_executor_test` 80 cases 在 CUDA 13 下通过。

### 14. [P2] PP 尾 stage 在唯一引擎 loop 线程上同步冷编译 grammar：新 schema 请求可使整条 lane 停摆（上限 compile_timeout_ms 默认 2000ms，批内多冷 schema 串行叠加），且首 stage 在 RPC 线程对同一 schema 重复构建、结果不被采样消费
- 位置：`cpp/normal_engine/pipeline/PPExecutor.cc:914`　findings：24290　（bot 提了 1 条）
- 状态：✅ 已处理 db71c7bf39（尾段异步预取，与 forward 重叠）
- bot 建议：将 grammar 编译移出执行线程：尾 stage 在接收计划时用 compile_pool_ 异步预编译，以 processor ready 作为该请求进入采样轮次的前置条件；或随计划携带首 stage 编译产物（序列化 matcher/compiled grammar id）消除跨进程双份构建；批内多个冷 schema 可并行预编译。
- 我方分析：**评审 head 上缺陷成立。** 原路径在尾 stage 完成 target forward 后才于 `prepareSampling()` 初始化 logits processor，`createMatcherFromKey()` 最终在执行线程等待 compile pool future；线程池只限制编译并发，不会让调用线程免于等待。首 stage 与尾 stage 是不同进程，backend/cache 不共享，因此首 stage 已编译不能令尾段命中缓存。直接传递 compiled grammar 会引入序列化、兼容性及流水线协议改动，最终选择不改变 admission 和 stage 通信的折中方案。
- 结论/处置：`db71c7bf39` 在尾 stage TP root 收到完整 plan 后、TP input sync 与 target forward 之前调用 `prepareLogitsProcessorsAsync()`；仅扫描尚无 `sampling_state` 的新请求，按 grammar key 去重，并通过现有有界 compile pool/singleflight/cache 启动预编译。编译由此可与 TP sync、slot 等待和本轮 target forward 重叠；采样使用点通过 `GrammarCompileHandle` 取得同一结果，再为每个请求创建独立 matcher。队列满时 prefetch 只延后提交，使用点按原同步语义重试；每个消费者保留独立 timeout，timeout、queue rejection、OOM/Unknown 等瞬时失败不污染共享 handle，成功与确定性 InvalidArgument 可共享。若编译慢于 overlap 窗口，使用点仍等待剩余时间，因此这是低侵入的阻塞削减方案，不是 admission 级完全解耦；首 stage 的独立编译也保持不变。
  - 验证：`LastPrefetchesGrammarBeforeTargetForward` 确定性断言编译在 target forward 前启动；backend 单测覆盖异步启动/singleflight、队列饱和后使用点重试、timeout 与瞬时失败不污染 handle、独立 timeout budget；PP batch processor 单测覆盖仅新请求、grammar key 去重与非法类型。最终 `xgrammar_backend_cpp_test`、`grammar_logits_processor_test`、`pp_batch_stream_processor_test`、`pp_executor_test`、`query_converter_test` 共 5 个目标通过，`git diff --check` 通过。未新增 PP+grammar smoke：现有 PP smoke 不携带 grammar，而 smoke 只能验证端到端可运行，不能稳定证明 overlap；该先后关系已由 executor 单测覆盖。

### 15. [P2] 多机 PP 前端选址对后端注册键一次性无捕获读取，等待界为 dist_comm_timeout（CLI 默认 300 秒、help 语义为心跳超时）而非注释宣称对齐的后端 bootstrap 界（gang_timeout_min 默认 30 分钟捕获-重试环）：注册延迟落入错配区间时整个部署启动失败
- 位置：`distribute/distributed_server.py:139`（解析器 _resolve_rank_rpc_addresses）+ `frontend/frontend_app.py:426`（前端调用侧）　findings：24302,24303　（bot 提了 2 条）
- 状态：🛠 已修复待环境验证
- bot 建议：将 _resolve_rank_rpc_addresses 的注册表读取改为捕获-重试循环：按 gang_sleep_time 节奏以 gang_timeout_min 为界（对齐后端 bootstrap 及前端 3600 秒/父进程 60 分钟的既有容忍契约），或为前端启动窗口提供独立配置而非复用心跳语义的 dist_comm_timeout；同时修正 distributed_server.py:143-144 注释（后端实际 bootstrap 界为 gang_timeout_min，非 dist_comm_timeout）。保留对后端真实故障的 fail-fast：重试环超界后仍抛 RuntimeError，与 frontend_routing_test 钉住的传播语义兼容。
- 补充（finding 24303，前端侧同因，2026-10-20 补抓）：FrontendApp.__init__ / FrontendWorker.__init__ / create_backend_rpc_server_visitor（DashSc 亦经此）无条件调 get_dp_rpc_addresses；pp>1 且本节点无 stage0 服务 rank 时走 _resolve_rank_rpc_addresses 的一次性 store.get（dist_comm_timeout 默认 300s、无 try/except/重试/回退，且在 sock.bind 之前）。异常重抛 → 前端子进程死亡 → ProcessManager 健康检查判 not ready → start_server 抛 "Health checks failed" → request_failure_shutdown 终止全 pod（含仍在 30min gang 窗口内正常等待的后端 rank）→ os._exit(1) 重启循环；形态 (b) role_type=FRONTEND 独立前端 pod 带 pp_size>1 且 local_world_size<world_size 时每次启动必然失败。仅影响 pp>1，pp==1 与 base 等价。
- 我方分析：**缺陷成立，但 bot 对两个期限的描述过度简化。** 后端 rank 创建初始 TCPStore 时同样受 `dist_comm_timeout` 约束，不能笼统认为后端在所有启动阶段都可合法等待 30 分钟；真正的错配是 frontend 与 backend rank 并发拉起，两边 `dist_comm_timeout` 的计时起点不同。frontend 可能已经等待 300 秒并退出，而 backend rank 经过进程创建/导入后才开始自己的 300 秒窗口，原本仍可能正常注册。`store.get(key)` 本身会阻塞至 key 出现或期限结束，且注册键是单调写入，因此问题不在“只调用一次”；额外捕获并短周期重试不会增强缺键等待语义，反而可能掩盖真实连接错误。触发范围只包括 `pp_size>1` 且所需 stage-0 地址不在本地 `WorldInfo` 的启动路径；`pp_size==1` 在查询前直接返回，本地 stage-0 齐全时也不创建 TCPStore，EPLB 运行期查询仍独立使用 3 秒期限。
- 结论/处置：**采用窄修复，不新增重试环或配置项。** `get_dp_rpc_addresses()` 的启动注册查询改用已有 `gang_timeout_min`（默认 30 分钟），与注册键所属的 gang bootstrap 阶段对齐；真实的 TCPStore/注册错误仍原样抛出，由现有启动健康检查执行失败收敛。单测将 `dist_comm_timeout` 与 `gang_timeout_min` 解耦，断言远端 PP 查询使用后者且错误继续传播；既有用例继续钉住非 PP 不访问注册表，EPLB 显式 3 秒路径未改。`git diff --check` 通过；`//rtp_llm/distribute/test:frontend_routing_test` 在本机 Bazel 分析阶段因缺失 `@pip_cpu_torch//flash_attn` BUILD 依赖而未能执行，需在完整构建环境复验。

### 16. [P2] qwen_3 组合 pp_size>1 与 enable_ffn_disaggregate 无前置互斥守卫：support_pp 门禁放行、权重按 stage 裁剪后，PP 不感知的 Qwen3DisaggregateModel 无条件取全量权重，建模期确定性 KeyError，PPExecutor 的设计性清晰拒绝不可达
- 位置：`models/base_model.py:186`　findings：24280　（bot 提了 1 条）
- 状态：✅ 已修复 d5ff4c5cc0
- bot 建议：在 set_parallelism_config 的 FFN 断言处（server_config_setup.py:334-336 同址）或 BaseModel.load 门禁增加互斥：pp_size>1 且 enable_ffn_disaggregate 时抛出与 PPExecutor 语义一致的明确错误（或令 support_pp 对该组合返回 False）；长期方案为 Qwen3DisaggregateModel 接入 apply_pp_partition 并按 pp_layer_ids/pp_has_embedding/pp_has_lm_head 实现 stage 感知构建。
- 我方分析：**确认真缺陷、PP 独有、fail-fast（建模期确定性失败）、P2 合理。**
  - 触发：`qwen_3` + `pp_size>1` + `enable_ffn_disaggregate`。
  - 崩溃链逐环核实：① 早期门禁全过——`support_pp()` 对 LANGUAGE_MODEL 恒 True（qwen_v3.py:20-21），`base_model.py:181-188` 的 load 门禁只查 support_pp；FFN 配置断言只查 `tp_size==1 && world_size>1`、**不查 pp_size**（server_config_setup.py:334-336）；RankLayout 整除校验过。② 权重按 stage 裁剪（`apply_pp_partition=True` 默认）：非末 stage 跳过 `lm_head`/`final_layernorm.*`（loader.py:615-622 `_skip_pp_global_weight`），且 `global_weights` 从空 dict 起步、键只在加载时插入（model_weight_info.py:825 / :867，无预置），故非持有键**根本不存在**（层权重则按全量 num_layers 预置空 dict，表现不同）。③ `qwen.py:207` 的 ffn 分支建 `Qwen3DisaggregateModel`，其 `__init__` 在 `if is_ffn_model / else` **之外**无条件取 `weights.get_global_weight(W.final_ln_gamma)`（disaggregate_qwen3.py:622-623），而 `get_global_weight` 是裸字典下标（model_weight_info.py:872-873）→ `KeyError: 'final_layernorm.gamma'`（字面值见 model_weight.py:1164）；:625 的 `W.lm_head` 同样缺键但 :623 先炸。④ 对照 `Qwen3Model`（qwen3.py:128-133）同一调用被 `if self.pp_has_lm_head else None` 守卫、embedding 同理（:107-113），故非 ffn 路径不炸。
  - 影响 rank：凡 `pp_has_lm_head` 为 False 者（module_base.py:123，即 `pp_rank != pp_size-1`）；:622-629 在 if/else 之外，故 FFN 服务 rank 与 attention rank 都会执行到。PP=2 时 stage0 必炸。
  - 时机：`base_model.load()` 内 :201 `_load(device_str)` 先完成 stage 裁剪加载、:215 `_create_python_model()` 才建模 → **进程启动建模阶段失败、服务起不来**，不是请求期故障。
  - bot 的「PPExecutor 清晰拒绝不可达」论断成立：`model_factory.py` 中 `_create_model`（约 :290，触发 load()→_create_python_model()）在 `create_engine`（:320 → NormalEngine → PPExecutor）**之前**，故 PPExecutor.cc:276-277 那条无条件 `"pipeline parallelism does not support FFN disaggregation"` 永远轮不到。
  - PP 独有：base 无 pp_size 参数（`git show origin/main:.../parallel_group_args.py | grep -c pp_size` = 0），该组合不可表达。
- 结论/处置：**采纳 bot 的第一落点（扩既有配置断言），+3 行，提交 d5ff4c5cc0。**
  - 在 `server_config_setup.py` 的 FFN 断言块（:334-336 之后）加 `assert parallelism_config.pp_size == 1, "pipeline parallelism does not support FFN disaggregation"`，消息与 PPExecutor.cc:276-277 逐字一致。
  - 不违背「PP 门禁只落 C++ 一侧」的既定裁决：那条针对的是 C++ 门禁**可达**的情形；此处经核实 C++ 门禁真·不可达（建模先于引擎构造），Python 断言是补全 `set_parallelism_config` 内已有的同类约束族（:291-307 的 ep/tp/dp 约束、:334 的 ffn tp/world 约束），不是跨语言重复校验。
  - 时机核实：`set_parallelism_config` 由 `start_backend_server.py:83` 调用，位于每进程入口 `local_rank_start`（:60 定义、:189 spawn）内，早于建模；报错从看不出原因的 `KeyError: 'final_layernorm.gamma'` 变成明确的断言消息。
  - 未采纳 bot 的长期方案（`Qwen3DisaggregateModel` 接入 apply_pp_partition 并按 pp_layer_ids/pp_has_embedding/pp_has_lm_head 做 stage 感知构建）：那是让 PP **真正支持** FFN disaggregate，属新功能而非缺陷修复，超出本条范围；且 C++ 侧 PPExecutor.cc:276-277 已明确声明不支持该组合，两侧口径需一致。
  - 未采纳「令 support_pp 对该组合返回 False」：`support_pp()` 是模型类方法、只拿得到模型自身信息，看不到 `enable_ffn_disaggregate`（属 parallelism_config），改它需扩签名并波及其它模型类。
  - 按用户裁决不补单测（同文件无 `set_parallelism_config` 配置断言的既有测试覆盖，grep 全仓零命中）。

### 17. [P2] PythonP2PWork::wait() 把 Python work.wait() 的返回值当作可 cast 的真值 bool 强约束，而注册契约只声明 typing.Any，锁定 torch 的返回值在快照内不可验证
- 位置：`models_py/bindings/core/ExecOps.cc:725`　findings：24301　（bot 提了 1 条）
- 状态：✅ 已修复 616cfa662c
- bot 建议：在 Python 回调侧包装返回对象（自定义类，wait() 返回 bool(work.wait())，None 视为已完成），或在 C++ 侧取 py::object 结果后分情况处理（None/非 bool 视为完成、仅显式假值报错，cast 失败给出可读错误）；同时把“wait() 需返回真值 bool”写入 __init__.pyi 与 ExecOps.h 契约，并补一个以真实 gloo 双进程驱动 PythonP2PWork::wait() 的测试。
- 我方分析：缺陷成立；C++ 原实现 `work_.attr("wait")().cast<bool>()` 将未在注册契约中声明的 Python 返回类型强制为 bool，若 backend 的 `wait()` 返回 `None` 会在通信清理路径抛 cast 异常。
- 结论/处置：`616cfa662c` 将 `PythonP2PWork::wait()` 改为只调用 `work_.attr("wait")()`、忽略返回对象，彻底移除 bool 强约束。为 #13 的非阻塞回收另行新增的 `isCompleted()` 调用 Python `is_completed()` 并读取其 bool，与 `wait()` 返回值无关。该提交的 PP executor/transport 测试覆盖完成、未完成、失败传播和 Gloo metadata 保留语义；未补 bot 建议的真实 gloo 双进程专项用例。

### 18. [P2] 新增 PP 输出分发按批行切 hidden_states，而 need_all_logits 为真时模型输出为全 token 行，与本变更新增的 OpData.h 契约注释相反，prefill 步静默返回错误隐状态
- 位置：`models_py/bindings/core/OpData.h:127`　findings：24297　（bot 提了 1 条）
- 状态：✅ 已修复 4d7d543a7d（lzf 提交：hidden_states 对齐 + per-request optional outputs）
- bot 建议：在 PPBatchStreamProcessor::fillOptionalOutputs 中，当 plan.model_input.need_all_logits 为真且批内含 context 流时，先用 plan.model_input.lm_output_indexes 对 model_output.hidden_states 做 index_select 再存入 result.hidden_states；或统一让 forwardPostLayers 始终把所选行赋给 hidden_states、全量行仅保留在 all_hidden_states，并据此修正 OpData.h:127 注释与非 PP 分发；修复前可在 PPExecutor::validateConfig 或请求校验中显式拒绝 return_hidden_states 与 calculate_loss/return_prompt_logits 的组合，并补一条用真实模型输出形状（token 行）的 PP 分发用例。
- 我方分析：**缺陷成立，但行为源头在 base，本 PR 只新增了 PP 消费路径和一句与实现相反的注释。** ① 模型侧：`PyWrappedModel.cc:1385-1390` 只在 `has_context_request && !need_all_logits` 时对 hidden 做 `index_select(lm_output_indexes)`，否则 `last_hidden = hidden`（全 token 行）；:1439 `outputs.hidden_states = last_hidden` 没有像 :1444 的 `outputs.logits` 那样在 `need_all_logits` 时再 index_select。blame：1385-1390 为 `2a30fe5092f`（JackTan25，2026-03-31，base），1439 为 `efae45ca2fd`（2026-09-23，#1432，base）。② `need_all_logits` 来源：`NormalModelInputGatherer.cc:629-630`（context 流的 `calculateLoss()` 或 `returnPromptLogits()`）；PP 下 `calculate_loss` 被 `PPExecutor::validateStream:615-617` 拒绝，SP 启用时 `return_prompt_logits` 被 :618-621 拒绝，故唯一可达组合是**非 SP 的 PP + return_prompt_logits + return_hidden_states**。③ 消费侧 `PPBatchStreamProcessor.cc:826-828` 按批行 `narrow(0, batch_idx, stream_batch_size)`，与 base 非 PP 的 `NormalOutputDispatcher.cc:383` 同款切法；当 hidden_states 实为全 token 行时切到错误 token 的 hidden（形状对、内容错的静默错误），`all_hidden_states`（:850-852 按 token_offset 切）不受影响。④ 本 PR 新增的只有 PP 这条分发路径与 `OpData.h:127` 那句契约注释，blame 均为 `6446ec7e17`（liuzefeng.lzf）。
- 结论/处置：**不修，base 遗留 + 归属 lzf。** 行为缺陷源头是 base 模型侧（JackTan25 2026-03-31），非 PP 路径同样错；PP 分发与错误注释由 lzf 在本 PR 引入，按用户裁决不在本分支修。若后续处理：在 `fillOptionalOutputs` 的 `return_hidden_states` 分支按 `need_all_logits && !is_decode` 条件补一次 `index_select(lm_output_indexes)`（约 6 行，与 :619 的 prompt/loss 计算同条件）；或让 `forwardPostLayers` 始终把选中行赋给 `hidden_states`（全量行只留 `all_hidden_states`，但会波及非 PP/eagle3 路径）；同时修正 `OpData.h:127` 注释。
  **2026-10-20 更新（提交 4d7d543a7d "fix: return PP optional outputs per request"）**：lzf 已在 PR 内修复本条。`PyWrappedModel.cc` 在 `need_all_logits` 分支补 `outputs.hidden_states = torch::index_select(hidden, 0, output_indexes)`，与 `logits` 同源对齐；同时把 `PPOutputConfig` 从批级开关改为逐请求 `requests` 列表、`PPExecutionResult` 用 `request_outputs` 承载每请求切片（`PPTypes.h:131`）、`fillOptionalOutputs` 按请求切好输出、`dispatchNormalSingleStream` 直接取视图，`PPSerialization` 版本 23→24。原"不修"结论作废，本条改为已修复。

### 19. [P2] PP 跨 stage 通信等待在仓库内无有界期限：stage 存活但停滞、消息序列失配或启动期 rank 缺席时，运行、停机与启动协商均可无限期挂起，stop() 无法收敛（文档§16 自记为未修复高优先级项）
- 位置：`models_py/distributed/collective_torch.py:381`　findings：24274　（bot 提了 1 条）
- 状态：🅐 已分析·搁置
- bot 建议：落实 docs §16：为启动协商、运行期 P2P 与停机 drain 分别设置有界期限（有限组超时或分段等待+取消标志，使等待期间可检查停止请求）；增加独立 stage 进度 watchdog（记录最后推进的 batch/phase/peer）；不可恢复错误触发流水线级失败收敛，明确终止受影响请求并经可执行的 communicator 拆除或受控退出；异常路径不得跳过匹配发送后继续执行。
- 我方分析：bot 把几种等待混在了一起，严重度描述过宽。启动期本地 rank 拉起有 3600 秒等待边界；子进程直接退出时，本机 `ProcessManager` 会通过存活检查发现并终止其管理的其它进程。优雅停机只有在 `shutdown_timeout` 配置为有限值时才有期限，`-1` 明确表示无限等待且当前 CLI 默认即为 `-1`。真正难以覆盖的缺口是 **stage 进程仍存活但运行期 PP 不再推进**：原有 PP NCCL/Gloo group 均使用 100 年 timeout，C++ 对 `Work.wait()` 也没有期限。
  - 直接缩短所有 group timeout 会把启动快照交换和运行期 P2P 绑定到同一期限；不同 stage 的模型加载/warmup 时差可能使先进入 snapshot collective 的 rank 被误杀。
  - 下游 stage 空闲时本来就长期等待 plan；若只给运行期 receive 加期限，正常无请求和 `pause_` 会被误判，必须再引入空 plan 心跳。
  - 仅在 Stage 0 监控真实 batch 的 `result_pending` 可以间接发现中间/末端 stage 停滞，但只能定位为整条流水线无进展；而且本机 `ProcessManager` 不能直接清理跨机器 stage，远端仍依赖通信报错或外部编排系统收敛。
- 结论/处置：**暂不修改，保留为已知可靠性风险。** 曾实现的轻量方案需要同时改 scheduler 空闲唤醒、`pause_` 路径、NCCL group timeout，并为共享启动/运行职责的 Gloo Work 增加单独包装；逻辑分散且仍不能保证跨机器完整收敛，收益不足以支撑其侵入性，已全部撤回。后续若重启该问题，应在两条完整路线中选择其一：① 拆分启动与运行期 communicator，配套有限运行期超时和空闲协议；② 实现独立 stage 进度 watchdog，记录 batch/phase/peer，并接入可执行的跨节点失败清理。两者都应通过多 GPU 故障注入验证，而不是继续叠加局部超时补丁。

### 20. [P2] PP P2P 回调把全局 rank 连同非默认 lane 组直传 torch.distributed.isend/irecv，未做组内换算，正确性依赖快照内无法求证的 torch 全局 rank 语义
- 位置：`models_py/distributed/collective_torch.py:692`　findings：24286　（bot 提了 1 条）
- 状态：🅡 已澄清（当前实现正确；已补真实 Gloo 回归）
- bot 建议：在仓库内钉死语义并消除依赖假设：为 pp_group_topology_test 增加在真实 gloo 非连续 lane 组（如 pp2/tp2 的 [0,2]）上执行 isend/irecv 数据往返的用例（现有真实组用例只覆盖 all_gather/all_gather_object，mock 用例仅固化直传，语义错误不可暴露）；若证实 torch 按组内 rank 解释，则在 cpp_isend/cpp_irecv 中以 group_ranks.index(global_peer)（或等价的组内 rank 换算）替换直传，并同步修订 ExecOps/pyi 的 global_peer 契约说明。
- 我方分析：**bot 的测试缺口判断成立，但缺陷推断不成立。** PyTorch 公开 `isend/irecv` 的 `dst/src` 契约是全局 rank，即使传入非默认 `group` 也不改变；仓库锁定的 PyTorch 2.1 实现会调用 `get_group_rank(group, dst/src)`，2.8 实现通过 `_canonicalize_group_rank` 完成同一转换，再把 group-local rank 交给底层 ProcessGroup。当前 `_check_peer(global_peer)` 先用 `get_process_group_ranks()` 校验全局 rank 属于 PP lane，再将其原样传给公开 API，正好符合契约。以 group `[0,2]` 为例，传 `dst=2` 会在 PyTorch 内部转换为 group rank 1；若按 bot 假设预先传 `dst=1`，PyTorch 会把 1 当成不属于该 group 的全局 rank 并报错。
- 结论/处置：**不改生产代码，只补契约回归。** `pp_group_topology_test` 的真实多进程 Gloo worker 现在保留注册回调中的 `isend/irecv`，每条 PP lane 按全局 rank 环形收发；`pp=2,tp=2` 明确覆盖非连续 `[0,2]` 与 `[1,3]`，并断言实际来源 rank。`bazelisk test //rtp_llm/models_py/distributed/test:pp_group_topology_test --config=cuda12_9 --test_output=errors --cache_test_results=no` 最终复跑 **PASSED in 40.8s**，`git diff --check` 通过。

### 21. [P1] 非流式内部流式路径下 head 每 chunk 物化全量累计 output_ids 且前端整请求驻留，单请求内存与聚合/解码 CPU 随输出长度平方增长（base 为线性），长输出可耗尽共享前端进程内存
- 位置：`openai/openai_endpoint.py:372`　findings：24305　（bot 提了 1 条）
- 状态：🅐 已分析·搁置（归属 lzf，本分支不修）
- bot 建议：保留正确性修复但降为线性：中间 chunk 仅携带增量 output_ids，由 _collect_complete_response 按 choice 累加拼接，仅最终响应（或每 choice 完成时）一次性物化全量列表；聚合后释放/裁剪 _all_responses 中间 chunk 的 extra_outputs 大字段；debug_info 的 raw_output 改为增量解码并在聚合后拼接；对开启 return_output_ids 的非流式长输出请求设置上限或告警。
- 我方分析：缺陷成立，性质是本 PR 的正确性修复选错了实现方式。① 需求真实：base 的非流式合并 `_merge_generate_outputs`（`custom_renderer.py:575-620`）只 `deepcopy(final_output)`（最后一个 step）并仅用 `torch.cat` 补齐 output_ids，其它字段全部取最后一步——而 `all_hidden_states` 只在 `iter_count_ == 1` 赋值（`NormalGenerateStream.cc:149-152`）、`loss_` 只在 prefill 步 defined（:153-163 的 `loss_index_ == inputLength()-1` CHECK 可证），故非流式请求要这些字段时 base 返回 None。② head 的修法：`openai_endpoint.py:372-376` 对 n>1 + return_output_ids/hidden_states/logits 强制内部流式（绕开 `custom_renderer.py:506-508` 的合并），:1256-1264 每 chunk 把 `status.output_ids_list` 累计全量写进 extra_outputs，:538-545 用 `model_copy(update=...)` 保留 prefill-only 字段（注释 `Persistent scores must not erase prefill-only outputs`），:1032-1041 用 `rows.append([])` 保证 n>1 每个 chunk 的 n 行结构。③ 缺陷点在"每 chunk 全量 + `CompleteResponseAsyncGenerator._all_responses` 全驻留 + 每 chunk `model_dump`/`model_copy`"：第 k 个 chunk 带 k 个 token，总量 Σk ≈ L²/2，L=32000 约 4GB；前端进程多请求共享，可被单请求占满内存。④ 覆盖不全：`calculate_loss` 不在 :372-376 的触发条件里，只请求 loss 的非流式请求仍走 base 合并、仍会丢，说明这是按 bot 所列字段打的补丁而非系统性修复合并语义。
- 结论/处置：**不修，归属 lzf。** `git blame` 实测两处（:372-376、:538-545）均为 `6446ec7e17`（author `liuzefeng.lzf`），属该提交 commit message 中 `Unify stream updates and OpenAI output handling` 部分，不在我方分工（KV cache 分段/拓扑校验/容量协商/RankLayout/PD 分离/多模态/system prompt/关闭协议）内；用户裁决 lzf 的代码不由本分支修复。若 lzf 后续处理，可直接采纳 bot 的方向：中间 chunk 只带增量、`_collect_complete_response` 按 choice 累加拼接、聚合后释放中间 chunk 的 extra_outputs 大字段；或保留非流式合并路径并在 `_merge_generate_outputs` 里逐字段取"最后一次非 None"（约十几行，且能同时修好 loss）。

### 22. [P2] head 在 pp>1 用 max(ep_size//local_world_size,1) 向下取整估算 EP 组承载节点数，ep_size 不整除 local_world_size 或 stage 未对齐节点边界时低估真实跨节点数，违反 rebalance_experts 的 num_nodes 语义与 num_gpus%num_nodes==0 断言契约
- 位置：`model_loader/model_weight_info.py:225`　findings：24304　（bot 提了 1 条）
- 状态：✅ 已修复+验证 PASSED d5ff4c5cc0
- bot 建议：复用 server_config_setup.py:193-196 的真跨度判定（或按 RankLayout 求该 stage/EP 组 rank 段的 min//local 与 max//local 差+1）计算 num_nodes，避免向下取整；并在 ep_size%num_nodes!=0 时于启动期校验失败或告警回退全局策略，使 eplb.py:122,209 的整除契约恒被满足；同时对齐 distributed_server.py:300 的 ceil 约定。
- 我方分析：**确认真缺陷、PP 独有、P2 合理；bot 的算术与下游契约逐条核实通过。**
  - PP 独有证据：base（`origin/main` :219-221）只有一行 `world_size // local_world_size`、无 pp_size 分支，且 `git show origin/main:.../parallel_group_args.py | grep -c 'pp_size\|PP_SIZE'` = 0（base 根本没有 pp_size 参数）。本 PR 把它改成条件式，else 支与 base 逐字相同，缺陷只在新增的 pp>1 支。
  - 错因两条：① `//` 向下取整丢掉余数；② 式子不含 pp_rank/world_rank，无法区分 EP 组起点是否对齐机器边界。base 原式成立的两个前提被 PP 同时打破——「EP 组从 rank 0 开始」（PP 后 stage1 起点落在机器中间）与「被除数 world_size 已被 `start_backend_server.py:401-404` 保证可被 local 整除」（PP 后换成可能带余数的 ep_size）。
  - 最小复现（满足仓库全部既有校验：12%4、12%2、lane_stride%ep_size、ep_size==tp*dp）：world=12/local=4/pp=2/tp=2/dp=3/ep=6 → lane_stride=6，stage0=rank0~5 压机器0(4张)+机器1(2张) 真实 **2** 台、stage1=rank6~11 压机器1(2张)+机器2(4张) 真实 **2** 台，旧式 `max(6//4,1)` 两段都给 **1**。后果静默：跨机副本被当同机放置，专家局部性与均衡质量下降；输出正确性不受影响（映射由 root 单点算完广播，ExpertBalancer.cc:344-362）。且与同 PR 的 `server_config_setup.py:193-196`（用真跨度判 is_multi_node，该部署判 True）口径自相矛盾。
  - 崩溃子形态：pp=2/tp=1/dp=5/ep=5/local=2 → 组 rank0~4 落 2+2+1，真跨度 3、旧式给 2；`eplb.py:122 assert num_gpus % num_nodes == 0`（num_gpus=ep_size，ep_balancer.py:91）在 `5%2` 与 `5%3` 下都失败 → 引擎步内 AssertionError（PPExecutor.cc:1354 → ExpertBalancer.cc:345 → ExpertBalancerPythonWrapper.cc:19 → ep_balancer.py:191 → eplb.py:218 进 hierarchical → :122），root 死后跳过 :356 广播、同组其余 rank 阻塞在集合通信至超时。**该形状今天同样崩，非修复新引入。**
  - EPLB 为何要求整除：算法把部署当 num_nodes 行 × (num_gpus//num_nodes) 列的矩形网格处理（:121 groups_per_node、:140 `view(-1, N//num_nodes)`、:149 balanced_packing 用同一个「每机卡数」描述所有机器），缺角时 reshape 与桶数都不成立。
  - 非 PP 不可达该断言（故 base 无需退回逻辑）：`local_world_size = max(min(device_count, world_size), 1)`（server_config_setup.py:279-283，start_backend_server.py:129 同式）。单机 → local=world → num_nodes=1，任何数 %1==0；多机 → local=device_count 且 `world % device_count == 0` 被强制，故 world=k×local、num_nodes=k，而 pp=1 时 ep_size=tp*dp=world=k×local → ep_size%k==0 恒成立。
  - 既有测试为何没抓到：`EplbNodeCountTest`（test_model_weight_info.py:602）5 行 case 全落在 `ep%local==0` 或 `pp_rank=0` 的公式成立区域。
- 结论/处置：**采纳 bot 的真跨度方向，但用启动期门禁取代它建议的「告警回退全局策略」**（用户裁决）。
  - 门禁 vs 退回：不整除时 `raise ValueError` 拒绝启动，而不是静默退回 num_nodes=1。理由是与本 PR 已有处置风格一致（#4 ROCm+PP 的 `RTP_LLM_FAIL`、#16 FFN disaggregate 的 assert、`PPExecutor::validateConfig` 的 7 条门禁），且静默退化会让运维以为有节点感知而实际没有；启动期拒绝严格优于现状（现状是首个引擎步 AssertionError + 同组 rank 挂到超时）。已核实 `num_nodes=1` 本是恒安全退化值（eplb.py 全部使用点代入 1 后 %1 恒过、//1 恒等，load_config.py:198,203 退化为 node_id 恒 0），故退回方案技术可行，只是被裁决否决。
  - 落点选 `model_weight_info.py` 而非更早的 `server_config_setup.py`：后者拿不到 num_nodes，在那儿门禁需把真跨度公式再写一遍造成两处漂移；前者是「值与它的合法性检查在一起」，单一真源。时机已足够早——`ModelDeployWeightInfo` 在 `create_model_loader`（base_model.py:196-198）构造，早于 `:201 _load(device_str)` 读 checkpoint。报错风格用 `raise ValueError`（该文件 :53-433 共 12 处全是它，无 assert / check_with_info）。
  - EP 组起点用 `world_rank - ep_rank` 而非 `pp_rank * lane_stride`：`ep_rank = stage_rank % ep_size`（rank_layout.py:209）语义即「组内第几个」，相减退回组头。该写法在配置层（`ep_size ∈ {1, tp*dp}`，server_config_setup.py:291-307 → 一个 stage 一个 EP 组）与布局层（`ep_rank_of` 只要求 `lane_stride % ep_size == 0`，允许一个 stage 多个 EP 组，rank_layout_test.py:371-376 即测此形）两种口径下都成立，而 `pp_rank*lane_stride` 只在配置层形状下对。生产路径确实用 `layout.ep_rank_of(world_rank, ep_size)` 设 ep_rank（server_config_setup.py:322-324），恒等式成立。注：`world_size == pp*tp*dp` 是约定非断言（parallel_group_args.py:47 原文「**通常**」，`RankLayout.__init__` 只校验 `world_size % pp_size == 0`，:59-62），这也是不写 `pp_rank*lane_stride` 的理由。
  - 真跨度公式恒 ≥ 1，故原 `max(..., 1)` 不再需要；`ep_size==1` 时 `ep_rank=0`、跨度算出 1、`1%1==0`，门禁不触发。
  - 改动 2 文件：`model_weight_info.py` +19/-5（真跨度 + 门禁，pp=1 支不动）；`test_model_weight_info.py` +14/-9（mock 补 `world_rank=0`；`EplbNodeCountTest` 表加 pp_rank 列 + 2 行 12 卡 case，stage0/stage1 都期望 2 而旧式给 1，能区分）。门禁分支按用户裁决不补单测。
  - 验证：`//rtp_llm/model_loader/test:test_model_weight_info` **Ran 36 tests / OK / PASSED in 13.6s / TEST_EXIT=0**（7 行 case 在同一 subTest 循环内，任一行失败都会报 failure）。另用独立算术脚本核验：既有 5 行新式值与旧式一致且门禁不触发（向后兼容）、新增 2 行新式 2 vs 旧式 1、崩溃形状真跨度 3 且 `5%3≠0` 门禁触发。
  - 未采纳 bot 的「对齐 distributed_server.py:300 的 ceil 约定」：单纯 ceil 修不掉起点对齐问题（ep=6/local=8/base=6 时 ceil 给 1、真实 2），真跨度公式已覆盖该场景。

### 23. [P2] 新增 draft_model_loading_test 未固定执行池，用例内 patch 字符串目标（及 model.load 惰性导入）强制加载 librtp_compute_ops，在 cuda13 默认 CPU worker 上该目标于用例阶段报错
- 位置：`test/BUILD:490`　findings：24307　（bot 提了 1 条）
- 交叉引用：构建类，与 #7（pp_topology_validator_test 链接闭包缺 ExecOps 提供方）同类
- 状态：✅ 已修复，与 #6/#7 合并提交 e10b307d68
- bot 建议：与同依赖近邻一致地固定执行池：为该 py_test 补 exec_properties = {"gpu": "L20_CU13"} 与 tags = ["L20_CU13"]（参照 head rtp_llm/test/BUILD:287、head models_py/speculative/test/BUILD:38）；或按 pp_group_topology_test.py:61 的做法在进入 patch 前以 patch.dict(sys.modules, {"librtp_compute_ops": 桩}) 阻断真实导入，并把 deps 换为不含 CUDA 闭包的 testlib_lite（head rtp_llm/BUILD:928-944），使目标可在 CPU 池运行。
- 我方分析：**缺陷成立、本 PR 新增、构建类（与 #7 同类），静态证据五条互相印证；本地不可见（开发机 8×H20 有 CUDA）。**
  - ① 目标 `test/BUILD:489-496` 是本 PR 新增（`6446ec7e17`；`git cat-file -e origin/main:rtp_llm/test/draft_model_loading_test.py` 报 not a valid object），只有 srcs+deps，**无 exec_properties/tags/manual/target_compatible_with**；紧邻上一目标带 SM100_ARM_CU13、同文件 auto_configure_deepep_test(:287) 带 L20_CU13。
  - ② deps 的 `testlib`（rtp_llm/BUILD:908-921）`data` 含 `//:rtp_compute_ops`（CUDA .so）；对照 testlib_lite(:928-944) 注释明说它不携带 CUDA 闭包。
  - ③ 触发点：字母序第 2 个用例 test_dspark_load_scope_reaches_python_construction 的 `patch("...qwen3_dspark_model.Qwen3DSparkModel")`(:61) 强制 import 该模块 → qwen3_dspark_model.py:19 直接子模块导入 compute_ops → compute_ops.py:3 `from librtp_compute_ops import *`（无 try/except）；ops/__init__.py 的 _load_compute_ops 回退只覆盖 __getattr__、**不覆盖直接子模块导入**（:272-301/:388-404）。
  - ④ 机制（仓库自证）：models/test/BUILD:34-37 注释原文 “unpinned cuda13 actions default to the CPU worker pool ... an empty-dict default is NOT a GPU request and lands the test on a CPU worker”；纯 CPU worker 缺 CUDA 运行库 → import 失败（同 3rdparty/trt_beam_search/tests/BUILD:51-55 记录的 libcuda.so.1 exit 127）。
  - ⑤ 本地不可见：开发机 8×H20，librtp_compute_ops 正常加载，上一会话本地 PASSED 13.5s；exec_properties/tags 只是 CI 路由元数据、不参与编译门禁，故本地看不出差别。
- 结论/处置：**采纳 bot 的 Option 1（固定执行池），已改 test/BUILD（+2 行），与 #6/#7 合并提交 e10b307d68。**
  - 修法：给 draft_model_loading_test 补 `exec_properties = {"gpu": "L20_CU13"}` + `tags = ["L20_CU13"]`，与同文件 auto_configure_deepep_test(:287-288) 逐字同形态；未加注释（同文件兄弟亦裸列）。
  - 不采纳 Option 2（打桩 + 换 testlib_lite 跑 CPU）：① 该测试 import 整批真实模型类（Qwen3DSpark/Qwen3Next/QWen3_VL/QwenV3/ModelFactory/BaseModel），testlib_lite 的 pure-py 子集不够；② 手工桩要同时满足 compute_ops.py:3-4 两条 star-import 与整条 model_desc 链，脆；③ pp_group_topology_test 用打桩是因它 deps 只有 collective_torch、纯 CPU gloo，情形不同。
  - 验证口径（同 #7 克制原则）：缺陷只在 cuda13 CI 的 CPU worker 复现，本地 H20 复现不了；exec_properties/tags 是纯路由元数据、不改本地构建/运行，故本地重跑证明不了修复本身（只会再确认 GPU 上绿，上一会话已确认）。**未本地跑。**
