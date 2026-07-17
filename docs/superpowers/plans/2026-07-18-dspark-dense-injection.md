# DSpark 1b 第一刀:注入 dense 化(搬垃圾保形状)+ 索引设备化

目标:把 1a 的"变长精确注入"改成 MTP 同款的"定长搬垃圾",消灭 decode 尾部的
全部 host 参与(D2H 同步、Python 建索引、变长形状),为后续 CUDA graph 捕获
draft forward 扫清模型侧障碍。本任务本身**不**开 graph(fail-fast 保留)。

基线:feature/dspark-g2-noncausal-attn @ 0c08de906(1a 全链路完成,E2E 已跑通)。

## 0. 现状(1a)哪里变长、哪里同步

- executor 尾部 `MtpBatchStreamProcessor::updateDecodePostDSparkDraftModelInput`:
  ① `transfer_done_event->synchronize()` + 读 accept_len_cpu/accept_tokens_cpu
  (每轮一次 D2H);② CPU 拼 combo(anchor 查表);③ host 循环建 aux 行号,
  `index_select` 出 sum(accept_len) 行 → **行数随 accept_len 变**;
  ④ dspark_ctx_lengths = accept_len(变)。
- 模型 `qwen3_dspark.py::_build_ctx_write_indices`:`prefix_lengths.tolist()` +
  Python 循环建 batch_indices/positions/page_indices(host 在环内)。

## 1. 安全性论证(已核,写进代码注释)

dense 方案:每轮把 verify 窗口**全部 k+1 行** aux 特征照单全注,窗口 =
[old_prefix, old_prefix + k + 1)(old_prefix = verify 用的 prefix)。

- 行 0..accept_len-1:committed 位置,特征正确 ✓
- 行 accept_len..k(被拒轨迹的垃圾特征):落在位置 [prefix_new, old_prefix+k+1),
  ⊆ 尾部块 forward 自己的查询窗口 [prefix_new, prefix_new+k+1)。
  propose() 内先注入后 block forward;块 forward 的 attention 路径
  (rope+append 每层先写后读)会用自己的 K/V **覆盖**这些位置,且块只对
  [0, prefix_new) 前缀 + 块内可见——垃圾在被覆盖前**没有任何读者** ✓
- 下一轮 dense 注入窗口右移,committed 部分永远被有效特征覆盖(no-rollback
  overwrite 同款不变量)。边界:accept_len=k+1 时垃圾行数为 0;prefix cache
  复用只会带走 committed 位置的有效特征 KV ✓

## 2. 改法

### 2a. 跨语言通道:窗口基点(小)
"窗口以 prefix 结尾"的旧约定表达不了 dense 窗口(old_prefix+k+1 ≠ 新 prefix)。
仿 dspark_ctx_lengths 原样加一条:
- OpData.h `GptModelInputs` + OpDefs.h/cc `PyModelInputs` 增 optional
  `dspark_ctx_starts`(int32 [B],定义时窗口 = [start, start+ctx),覆盖
  "以 prefix 结尾"默认);PyWrappedModel.cc 透传(搜 dspark_ctx_lengths 三处照抄)。

### 2b. executor 尾部全设备化
`updateDecodePostDSparkDraftModelInput` 重写:
- anchor 设备 gather:accept_tokens(GPU [B,k+1]).gather(1, accept_len_gpu-1)
  → combo_2d = full([B,k+1], mask_id, cuda);combo_2d[:,0]=anchors
  (照抄 draftModelDecode 里 pre_target_device_gather 的写法)。
  **删掉 transfer_done_event->synchronize() 和 accept_*_cpu 依赖**。
- aux 不做 gather:last_hidden_states = aux.reshape({B*(k+1), -1}) 整块透传
  (MTP all_hidden_states 同款);hidden_states_d_t 同。
- dspark_ctx_starts = 旧 prefix(在 prefix_lengths += accept_len **之前** clone);
  dspark_ctx_lengths = full({B}, k+1)。
- lm_output_indexes/input_lengths 不变。
- 注意:accept_len_cpu 在 dispatch/bookkeeping(prepareDecodeSpecUpdateInfo)
  仍要用,那条路不动——本任务只消 executor 尾部这个同步点。

### 2c. 模型注入设备化
`inject_context_kv` 增设备快路(ctx_starts 给定时):
- batch_indices = repeat_interleave(arange(B), k+1);
  positions = (ctx_starts[:,None] + arange(k+1)).flatten() —— 纯 device,定形。
- **绕开 flashinfer append 的 ragged 页表接口**(建 page_indptr 需 host 总数
  = 又一次同步):直接散写。物理槽 = block_table_device[i, pos//page]*page
  + pos%page;kv_cache_base [P,2,nkv,page,hd] 用高级索引
  `base[page_idx, 0, :, slot, :] = k_rows`(K/V 各一次)。纯 torch、定形、
  无 flashinfer 依赖。
- 旧变长路径(ctx_lengths 语义)保留给 prefill 播种(prompt 后缀天然变长,
  prefill 不在 graph 收益面)和 golden UT 的增量用例。

### 2d. UT
- 散写 vs flashinfer append 布局等价:同一批 K/V 两种写法,cache 逐位相等。
- dense 注入等价:dense(k+1 行含垃圾)注入 + 块 forward 的 draft tokens ==
  增量注入(1a 路径)的结果逐位一致(垃圾被覆盖的实证)。
- 既有 golden 9 条 + capture 2 条保持全绿。

## 3. 验收 gate

- E2E:重跑 build_logs/dspark_gate/e2e_dspark_gate.py 的 dspark 侧,输出与
  1a 的存档 build_logs/dspark_gate/dspark.json **逐字相同**(committed 行特征
  值不变、块 forward 数学不变,理应比特级一致——不一致即有伤)。
- 若碰了 MtpBatchStreamProcessor 共享段:eagle_mtp_tp2 远程回归
  (记得 --run_under gpu_lock)。
- 顺手量化:尾部播种段延迟(sp propose_step_latency_us 已接线)1a vs 1b。

## 环境备忘(照抄 1a 经验)
- bazel py 测试:--test_env=LD_PRELOAD=/usr/lib64/libnvidia-ml.so.1 +
  PYTHONNOUSERSITE=1;源码跑 server 相反:**不能** PYTHONNOUSERSITE,且
  PYTHONPATH 前置 bazel 侧 rtp_kernel wheel(见 memory 环境坑)。
- E2E 模型对:/data0/caihaowen.chw/dspark-work/models/{sft_model,dspark_draft_rtp}。
- 远程 smoke 必带 gpu_lock;/data0 常年逼近满盘,写大文件前 df 一眼。

## 非目标(别顺手做)
- 不开 ENABLE_CUDA_GRAPH(graph 还差 flashinfer plan 捕获、metadata prepare
  等,另立任务);不动 TP;不动 prefill 播种路径;不做动态 k/confidence(phase-2)。
