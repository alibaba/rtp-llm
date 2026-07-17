# DSpark phase-1a 收尾:MtpExecutor 缝合 + E2E greedy gate

前置(全部已完成并 commit,分支 feature/dspark-g2-noncausal-attn):
- 模型本体 `qwen3_dspark.py`(golden UT 8/8,draft tokens 逐位一致)
- G1 aux_hidden_states 出口(PyModelOutputs/GptModelOutputs/PyWrappedModel 三路径 + Qwen3Model 捕获,UT 2/2)
- G3 出口契约 draft_tokens/draft_probs(同三处 + 模型 forward 已填充)
- 配置面 SP_TYPE_DSPARK + --sp_dspark_propose_num + model_factory 接线
  (_setup_dspark_configs:k 写回 gen_num_per_cycle、target capture ids=[j-1])

本文剩余工作按依赖序排列。行号锚 pr-1107 基线 + 本分支(MtpExecutor.cc 未动)。

## 0. 侦察补课(动手前必读)

- [ ] `MtpBatchStreamProcessor.cc` 全文(930 行):重点
      `updatePrefillPostDraftModelInput`(:594-631 MTP 左移一位)、
      `prepareOneStepSpecDecodeModelInput` / `prepareDecodeDraftModelInput`、
      `updateDecodePostDraftModelInput`、`dispatchPrefill` / dispatch 系列
      (stream 侧 propose token 存储宽度:MTP 存 1 列,dspark 要 k 列)。
- [ ] `MtpExecutor.cc:335-523` 模型构造段:`sp_prefill_draft_model_` 建法、
      FastTopKSampler/SpeculativeSampler 从哪里 set(引擎 init 侧,
      RtpLLMOp/NormalEngine 一带)。
- [ ] `CacheConfigCreator.cc:70-223 createSpConfig`:确认 5 层 qwen3 draft
      (8kv×128hd)的 sub-config 正确(MTP 惯例是 1 层 mtp module;dspark
      draft 层数/几何不同,共池 block 字节 = 主 + 5×draft 层)。
- [ ] `ProposeModelEngineInitParams.h` + RtpLLMOp:sp_type 如何到达
      MtpExecutor(dspark 分支判据用它,不要用 propose_step>1)。

## 1. PyModelInputs 增 ctx_lengths 通道(小)

decode 尾部播种是**增量注入**:input_hiddens 只带本轮 accepted tokens 的
aux 特征,注入窗口 = [prefix - accept_len_i, prefix)。模型侧
`inject_context_kv(ctx_lengths=...)` 已留参数;缺的是跨语言通道:

- [ ] OpDefs.h `PyModelInputs` 增 optional `torch::Tensor dspark_ctx_lengths`
      (int32 [B],undefined = 注入整个 prefix,即播种语义)+ OpDefs.cc
      def_readwrite + PyWrappedModel 从 GptModelInputs 透传(GptModelInputs
      也要加同名字段,OpData.h)。
- [ ] `Qwen3DSparkModel.propose()` 读 `inputs.dspark_ctx_lengths`(定义时
      覆盖 ctx_lengths 参数默认)。UT:golden 测试加一条增量注入用例
      (先注 0..95,再以 ctx_lengths=[1] 注第 96 位,校验 cache 第 96 行)。

## 2. MtpExecutor 缝合(核心)

执行器侧 dspark 分支判据:ctor 存 `is_dspark_`(来自 propose_params sp_type)。

### 2a. 构造 (:335-523)
- [ ] dspark:不建 `sp_prefill_draft_model_`(块 forward 播种/decode 同形状,
      单对象);不建 FastTopKSampler(模型自带采样);SpeculativeSampler 照建
      (宽度构造参数 = k,gen_num_per_cycle 已=k);`propose_step_ = k`。
- [ ] 1a fail-fast:dspark × (异步四开关任一开 / PD role != PDFUSION / CP /
      ENABLE_CUDA_GRAPH) → init 报错;phase-2 flags(confidence/sts/sps 系列,
      见 dspark-phase2-design §API)注册即报错也放这批。

### 2b. prefillStep 播种 (:754-781 target sample 之后)
- [ ] `updatePrefillPostDraftModelInput` 的 dspark 变体(新方法,放
      MtpBatchStreamProcessor):
      - combo_tokens = 每 stream [anchor(=target 采样 token) + k×mask_id],
        input_lengths = k+1,prefix_lengths = prompt_len(context KV 已由
        target... 注意:draft 的 KV 池是自己的 group,prefix 特征 KV 由模型
        注入 pass 写,attention metadata 的 prefix_lengths 就是 prompt_len);
      - last_hidden_states = model_output.aux_hidden_states
        (PyWrappedModel 入口把它塞 PyModelInputs.input_hiddens,MTP 同款
        通道 :766 附近 `model_input.last_hidden_states = ...` 的 dspark 源);
      - dspark_ctx_lengths 不设(播种=全 prefix)。
- [ ] draft forward 返回的 GptModelOutputs.draft_tokens/draft_probs 直接作
      draft_sampler_output(跳过 fast_topk_sampler_ 段 :781-786);
      draft_probs 需 reshape/dtype 对齐 SpeculativeSampler 期望
      (读 SpeculativeSampler.cc:96-126 确认 all_probs 形状 [B,k,V])。
- [ ] `dispatchPrefill`:stream 存 k 列 propose tokens(现状 1 列;宽度改
      构造参数或按 propose_step_ 通用化——MTP propose_step=1 时兼容)。

### 2c. decodeStep (:930-955, :1053-1068)
- [ ] gather 变体(缝合点 4):verify 输入 = 每 stream [anchor, k propose
      tokens] 矩形 [B, k+1];`prepareOneStepSpecDecodeModelInput` 与
      `prepareDecodeDraftModelInput` 都不适用(前者 1 列,后者带 MTP 多步
      语义),新 `prepareDSparkVerifyModelInput`,`setVerifyPairInputs`
      原样复用。
- [ ] 跳过 draftModelDecode:`if (propose_step_ > 1 && !is_dspark_)`。
- [ ] `updateOneStep/MultiStepDraftSamplerOutput` 的 dspark 变体:
      draft_sampler_output 直接来自 stream 存的上轮 draft_probs
      (跨步存储:draft_probs [B,k,V] 挂 stream device state,写(t)/读(t+1),
      1a 同步时序单 buffer 即可)。
- [ ] 尾部播种(:1053-1068 runDraftPrefillForward 前):
      `updateDecodePostDraftModelInput` 的 dspark 变体——
      - anchor = 每 stream accept_tokens 最后一个(bonus/corrected);
      - combo_tokens = [anchor + k×mask],input_lengths=k+1,
        prefix_lengths = 新 committed_len;
      - last_hidden_states = verify model_output.aux_hidden_states 中
        **accepted 位置的行**(按 accept_len gather;verify 是 [B,k+1] 个
        token 的 aux,取前 accept_len_i 行——刻意含上轮 rejected 槽位的覆盖
        写由注入 pass 位置寻址天然保证);
      - dspark_ctx_lengths = accept_len。
- [ ] 尾部 draft forward 后:draft_tokens/probs 存 stream(下一轮 verify 用),
      跳 fast_topk(:1060 附近)。

### 2d. 观测(顺手)
- [ ] 接上恒 0 的 `sp_propose/score/sampler_step_latency_us`。

## 3. E2E greedy gate

- [ ] 需要 target ckpt(verifier=/data/oss_bucket_0/sft_model,Qwen3 dense
      5120/64 层?先确认本地有无;没有就用小 qwen3 + 假 draft 走烟囱:
      转换脚本可给任意 qwen3 造 mini-dspark ckpt——fc/markov 随机——只验管线
      不验精度,greedy 一致性靠"draft 全拒也得逐 token 一致"成立)。
- [ ] gate:`--sp_type dspark` on/off greedy 逐 token 一致(eager/sync);
      MTP 回归:`--sp_type eagle` smoke 全套(碰了 MtpExecutor 的硬性纪律)。

## 已知坑
- PyWrappedModel::forwardPostLayers 会对 draft hidden 再跑一次 lm_head
  (MTP 需要,dspark 冗余)——1a 容忍,挂 1b 优化(skip 开关)。
- pr-1107 基线 decode/cuda_graph 邻居测试本来就挂(fixture 传 None),
  与本分支无关,PR 描述里说明。
- bazel py 测试:--test_env LD_PRELOAD=/usr/lib64/libnvidia-ml.so.1 +
  PYTHONNOUSERSITE=1;deps 要 //rtp_llm:models + //rtp_llm:async_model。
