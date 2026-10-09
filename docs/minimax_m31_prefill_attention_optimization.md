# MiniMax-M3.1 最新BS60 Zero-CTA prefill attention分析

Trace: /data0/ruixuan.zrx/profiling_results/m31_cp4_same_layer_ag_bs40_60_80k_20261009/zero_cta/bs60

本文上半部分记录该campaign的冻结源码和历史测量，不代表当前在线容器仍加载同一份package。新实现、测试环境及未完成验证见“逐项实施记录”。

## 当前交付状态（模型 campaign）

- 第一批：已完成本轮新基线性能、四rank timeline、同ID 660条质量；pureBS40/60 native耗时下降2.65%/3.16%。主树保存第一批WIP，模型用独立冻结包。
- FI direct：初版性能退化已修复。fi_restore_stage1完整20轮、四rank trace、660质量通过；pureBS40/60相对第一批改善1.38%/0.73%，自然PD40/60改善1.63%/0.62%。GSM436/500，LongBench66.81715（5条truncation），单次分数差不作因果精度提升结论。rank0 restore138.076→64.046ms，converter60→0；writer仍78.110ms。四文件增量已集成主树，保留两处既有Decode阈值改动及全部其他WIP。模型冻结包未改。
- Score-fill：单算子真实OnlyScore/TopK通过。新容器score_recheck_stage1完成固定五场景20轮且零错误、四rank trace和660质量。pureBS40/60相对stage1中位耗时改善0.2855%/0.5082%，自然PD40/60改善约0.23%/0.35%；rank0 trace fill240→0、kernel4271→4031，但target2559.459→2561.504ms，没有明显总耗时改善。GSM435→434/500，LongBench66.61455→66.37585（各5条truncation）；单次差异不能因果归于fill删除，也不能称质量完全一致。候选不合入主树，完整失败/残留缓存失效补测保留。
- 调度/wire最终：direct/async未产生新CE overlap，保持后置。R16 wire完整五场景20轮960请求零错误、四rank timeline和660质量已完成，relative FI耗时下降1.14%–2.46%，pureBS40/60相对fresh baseline下降6.35%/6.23%。GSM436相同，LongBench66.817→66.238（4条输出变化，精度未签收）；主树仅默认OFF实验路径，稳定FI默认保留。四rank restore/suffix CE交叠均0。

完整数据与身份边界见文末“模型接入与后续并行批次”，campaign路径为 `/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration`。

统计每rank最后一次 executor.mtp.prefill_step(target_model_forward)，排除第一轮、draft和中间等待。GPU时间为duration sum；同时给union，不能跨stream或跨rank简单相加。

## 对上一版结论的修订

- 旧BS20/74011/reuse66560与新BS60/80000/reuse75776不构成性能A/B。新case新增253440 IDs，CP每rank63360 Q，四个index chunks为16384×3+14208。目录目前只有bs60 trace。
- Target 60层全部MSA；FFN前三层dense、后57层MoE。
- 主attention已改为FlashInfer vendor fused kernel，每层一次；不再有旧CSR五步、180次独立main/Combine，不能继续把Combine当当前优化对象。
- FlashInfer主路径是BF16 Q、packed KV4，选中tile在片上反量化成BF16做MMA及softmax；不是旧主Q8KV4 MMA，也没有全历史BF16扩张buffer。IndexScore仍为Q8/K8。
- native IndexScore compact rescue现在实际运行：全pool index stage37500 pages×16384 bytes=585.94MiB，单stage已超过512MiB模块预算。claim/stage_compact/remap每层四次；不是新代码异常。
- 同层prefix pack/AG/restore已在side stream39与main stream7 projection/norm重叠；next-layer prefetch仍OFF。Zero-CTA使用CE copy，不能仅用NCCL kernel计数判断通信消失。

## 每rank最后target统计

|arm|rank|target ms|kernel sum ms|kernel union ms|kernel+memcpy+memset union ms|<10us count|<10us sum ms|
|---|---|---:|---:|---:|---:|---:|---:|
|baseline|prefill_wr0_1|2822.682|2786.566|2786.431|2786.869|2064|6.074|
|baseline|prefill_wr1_1|2822.862|2787.143|2786.993|2787.467|2061|6.151|
|baseline|prefill_wr2_1|2823.100|2783.639|2783.504|2783.976|2061|5.857|
|baseline|prefill_wr3_1|2822.651|2784.471|2784.338|2784.761|2061|5.841|
|ordinary|prefill_wr0_1|2794.823|2871.158|2757.116|2757.615|2063|6.059|
|ordinary|prefill_wr1_1|2794.723|2874.049|2757.876|2758.375|2061|6.203|
|ordinary|prefill_wr2_1|2794.698|2869.559|2756.736|2757.208|2061|5.879|
|ordinary|prefill_wr3_1|2795.199|2863.318|2748.959|2749.430|2061|5.830|
|zero_cta|prefill_wr0_1|2613.693|2523.908|2522.691|2570.939|2064|6.139|
|zero_cta|prefill_wr1_1|2613.815|2518.233|2517.009|2563.506|2061|6.062|
|zero_cta|prefill_wr2_1|2613.658|2527.126|2525.940|2572.016|2060|5.927|
|zero_cta|prefill_wr3_1|2613.689|2515.417|2514.200|2561.918|2061|5.815|

rank0 latest target2613.693ms，GPU activity union2570.939ms，覆盖98.36%。<10us 2064kernel执行6.139ms，0.235%。publication CPU wait490.366ms，其中GPU activity覆盖487.564ms；不能把整个wait作为额外GPU idle。

## 数据流与逐kernel结果

每层side：prefix pack → prefix side/main两次CE allgather → restore；main：输入融合norm/quant → QKV/index QK投影 → Gemma norm/RoPE（保留BF16 Q）→ BF16 Q rows contiguous、suffix KV pack → side suffix CE AG → consumer等待prefix/ suffix完成 → suffix NVFP4写working及owner persistent → 四个index chunks compact staging/OnlyScore/copy/TopK → FlashInfer planar KV转换/tail清理/page table/TopK整理 → 一次BF16Q/KV4 fused sparse attention → MXFP8输出量化、scale pack和o_proj → FFN。

所有97个kernel名称在zero_cta_rank0_operator_inventory.csv中逐项保留次数、sum/median/min/max、用途及CPU parent/Input Dims。CSV仅将有证据的操作映射到具体语义；模板变体、高级索引不能靠名称猜测Python表达式。

|rank0算子|次数|sum ms|median us|
|---|---:|---:|---:|
|void deep_gemm::sm100_nvfp4_nvfp4_mega_moe_impl<16512u, 6144u, 3072u, 128u, 4u, 4u, 192u, 128u, 256u, 48u, 256|228|751.665|3230.046|
|void flashinfer::msa_prefill_nvfp4::(anonymous namespace)::sparse_prefill_kernel<flashinfer::msa_prefill_nvfp4|60|413.593|6756.993|
|void cutlass::device_kernel<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<cute::tuple<cutlass::f|240|242.966|1008.250|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 98|60|164.966|2813.323|
|_topk_to_block_table_multirow_kernel|240|134.032|567.221|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 61|60|107.280|1806.641|
|_restore_prefix_planes|60|85.211|1414.285|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 61|57|78.299|1389.773|
|_quantize_main_index_rows_d128_kernel|60|73.272|1208.204|
|_fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel|117|63.709|541.030|
|_gemma_norm_rope|60|62.938|1043.226|
|_convert_page_head|60|49.036|817.080|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 61|57|41.081|730.023|
|_pack_nvfp4_inputs_vector_kernel|228|29.388|131.617|
|_prefill_bf16_router|57|27.839|483.013|
|_copy_index_scores|240|25.001|106.273|
|void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*|57|19.694|344.995|
|_rows_to_contig_kernel|60|18.928|314.740|
|kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorpt|60|15.263|253.715|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 24|3|14.957|5025.040|
|_silu_and_mul_mxfp8_quant_tiled_kernel|60|14.215|204.930|
|void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>,|60|11.352|188.370|
|_stage_compact_index_pages|240|9.693|40.800|
|_row_gsf_kernel|228|8.689|39.200|
|void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 32u, 32u, 32u, 0u, 61|3|7.655|2552.792|
|_pack_prefix_pools|60|7.444|123.570|
|void rtp_llm::group_idx_and_topk_idx_kernel<float, long>(float*, float const*, float*, long*, float*, long, lo|57|6.326|109.665|
|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_|120|6.028|47.728|
|void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int|240|5.548|23.488|
|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10|10|4.958|490.437|
|void deep_gemm::transpose_and_pack_fp32_into_ue8m0<512u, 48u, 192u, 128u, 1u, false>(float*, unsigned int*, un|120|2.820|23.569|
|void deep_gemm::transpose_and_pack_fp32_into_ue8m0<512u, 48u, 96u, 128u, 1u, false>(float*, unsigned int*, uns|57|1.732|30.048|
|ncclDevKernel_AllReduce_Sum_u64_RING_LL(ncclDevKernelArgsStorage<4096ul>)|1|1.642|1642.256|
|_scale1_query_fp8_kernel|240|1.473|6.176|
|flashinfer::plan_kernel(int*, int*, int*, unsigned long*, unsigned long*, int, int, int, int, int, bool, int*,|4|1.472|380.100|
|_fused_add_rmsnorm_fp8_quant_singlepass_kernel|3|1.336|444.388|
|void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la|57|1.297|22.624|
|_pack_flashinfer_mxfp8_scale_kernel|60|1.142|18.848|
|rtp_llm::(anonymous namespace)::buildAttentionInputMetadataKernel(int const*, int const*, int*, int*, int*, in|1|1.130|1129.739|
|void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo|57|1.112|19.424|
|_prepare_topk|60|0.897|14.896|
|_claim_index_pages|240|0.783|3.232|
|void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, |481|0.729|1.568|
|void rtp_llm::topk_with_k2_kernel<float>(float*, float*, long, long, long, long)|57|0.656|11.360|
|_remap_index_pages|240|0.636|2.624|
|void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*|1|0.620|620.262|
|void at::native::vectorized_elementwise_kernel<4, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&)::{l|57|0.476|8.193|
|void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<long, long, long, at::native::bina|126|0.325|2.688|
|void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<int, |61|0.320|5.280|
|void rtp_llm::embedding_lookup_kernel_vec<float4, __nv_bfloat16, false, false, false>(__nv_bfloat16*, __nv_bfl|1|0.305|305.443|
|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at|60|0.229|3.792|
|void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1u|2|0.198|99.121|
|void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<long, long, long, at::native::rema|64|0.172|2.704|
|void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul> >(i|60|0.127|2.128|
|void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<long, long, long, at::native::bina|62|0.120|1.952|
|void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul> >(|62|0.118|1.888|
|_clear_packed_working_tail_scales_kernel|60|0.115|1.889|
|nvjet_sm103_tss_384x64_64x3_4x1_v_bz_TNT|1|0.114|114.497|
|void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2|63|0.112|1.824|
|_clear_unwritten_page_tail|60|0.108|1.760|
|void deep_gemm::transpose_and_pack_fp32_into_ue8m0<512u, 48u, 384u, 128u, 1u, false>(float*, unsigned int*, un|3|0.105|35.073|

## 当前优先级

IndexScore OnlyScore242.966+score copy25.001+TopK134.032=402.000ms，占target15.38%，与FI主attention413.593ms相当。可评估避免score整块转置、TopK直接消费native layout，stride效率和mask语义需验证。

FlashInfer _convert_page_head49.036ms，遍历37500 working pages×4KV heads，重排packed K/V及scale字节；每page73728bytes planar pool，额外2.575GiB/rank共享跨层。不是TopK只选中页转换。可研究writer/restore直接写FI layout或按活跃/选中页转换，需新ABI、owner及别名生命周期证明。

_rows_to_contig_kernel18.928ms复制BF16主Q；可研究norm/RoPE直接输出contiguous BF16 Q，不应沿用旧直接Q8输出的建议。

主FI sparse413.593ms：一次每层融合softmax/output，无独立Combine；需要单算子hardware counters及matched shape对照评估是否接近最佳吞吐，不能从kernel大就判慢。

prefix restore85.211ms虽在side stream仍几乎完全暴露，是当前关键通信后处理；suffix writer73.272ms处理全CP gathered suffix并写persistent+working，必要的量化与存储。side stream不会自动将restore隐藏。

compact stage9.693、claim0.783、remap0.636ms，相对index打分和TopK很小。强制allpool会超过现有scratch预算，不应为了删kernel直接扩大workspace。score fill5.548ms、TopK prepare0.897ms、两类tail清理共0.223ms，低优先级。

MoE mega751.665ms占target28.76%，是全模型最大单项；attention优化的总收益还受FFN约束。普通overlap会与projection/norm争用SM或HBM：rank0norm baseline60.645ms→ordinary75.114ms，ZeroCTA62.938ms；QKV baseline144.136→ZeroCTA164.966ms也没有保持不变，不能直接sum各阶段缩短预测wall收益。

## 同workload对照

|arm|rank0 profiled target ms|3warm rounds native model median s|
|---|---:|---:|
|baseline|2822.682|2.865539|
|ordinary overlap|2794.823|2.828695|
|ZeroCTA overlap|2613.693|2.662690|

匹配trace ZeroCTA相对baseline约7.40%缩短、相对ordinary约6.48%。无profiler Native model三warm-round中位数分别约7.08%、5.87%缩短；两种口径分别报告，不混作同一个结果。正式数据读取campaign performance_comparison.json，不代表本次重跑，亦不自动证明数值/质量接受。

以上数据来自既有 offline trace/source 分析；后续实现和验证单独记录，不与历史结果混用。


## 当前runtime源码证据

运行根目录为 `/data0/ruixuan.zrx/minimax-m31-dev/1009_same_layer_ag_pd40_60/prefill_install/rtp_llm`，不是旧operator目录。当前same-layer/review overlays及六个native so hash已核对匹配launch/contract；源码HEAD为d31fc89984c0bbc1615d4ef74e463afd0b223d7f且dirty WIP preserved。profile前各rank实际记录allocator=ncclMemAlloc、cta_policy=2、symmetric=True。

- msa_attention.py:4423–4437/4707：side stream启动当前层prefix；首层需要等待metadata，后续层可提前发。
- msa_attention.py:3761–3797：side AG → values AG → prefix restore，随后prefix-ready event。
- msa_attention.py:3810–3828：同一side stream等待suffix producer，然后suffix AG；不是prefix/suffix网络并发。
- msa_attention.py:3037–3043：writer消费前等待suffix-ready；working planes及通信slots用event生命周期保护。
- collective_torch.py:349–354和cp_same_layer_workspace.py:119–148：TP_SIDE policy2、symmetric NCCL registered MemPool与steady event fence；并非仅设置env变量。
- msa_attention.py:4758/4838/5049：FI不输出主Q8，转contiguous BF16 Q并调用FI adapter。
- flashinfer_nvfp4_prefill.py:124：全部page/head转换；:139–145：page table及TopK sort；:157–162：fused FI执行。
- flashinfer_vendor/csrc/msa_prefill_nvfp4_specialized.cu:592/665–672：selected tiles FP4→BF16；:408/419 MMA kind::f16，:1018 softmax P BF16。
- native_q8_index_score.py:83–128/434–487：512MiB预算与compact执行。该文件/topk_bt_fused.py与1008 candidate hash相同；行为变化来自geometry与backend选择。

## 同层通信：新的性能结论

同layer、最后target、四rank evidence：ordinary侧流NCCL累计318.2–326.8ms，与main kernel重叠112.5–115.9ms（约35%）；ZeroCTA CE memcpy累计241.9–243.4ms，与main重叠195.5–196.8ms（约81%）。这只是实际copy/kernel区间重叠，不包含所有NCCL annotation/semaphore等待。

Prefix restore仍累计84.4–85.3ms。ordinary与main overlap为0，ZeroCTA仅0.889–0.912ms，即约1%；把restore移到side stream并没有让它完全被掩盖。

ordinary prefix pack结束到第一prefix NCCL GPU执行开始的中位间隔2.38–2.47ms；ZeroCTA只有12–27us。典型ordinary layer30这段间隔接近当层QKV GEMM2.488ms，NCCL到GEMM末尾才开始；CE却能够与GEMM早期重叠。可确认NCCL GPU启动存在明显延后；CTA资源约束是否是唯一原因仍需scheduler/硬件计数验证，不能只从timeline判定。

rank0各层pack→suffix writer窗口中，main stream未运行kernel的累计时间：ordinary299.995ms，ZeroCTA147.493ms；writer前最后连续无main-kernel区间：ordinary291.374ms，ZeroCTA55.340ms。这些是主流空档，不是整GPU idle：side stream此时仍在restore/copy/通信，不能与GPU全设备union差值混淆。未来优先分析未盖住的restore及消费者边界，而不是继续假设prefix AG全部串行暴露。


## 进一步核实与修正（用户问题1–6）

Trace实际working capacity37560pages，而37500是60×80000/128的有效历史页数。FI pool实际容量37560×73728=2.579GiB/rank；全pool index stage实际586.875MiB，仍超过512MiB。之前约2.575GiB是按有效页计算的估计。

Norm后主流copy等待restore不能归因显式early prefix wait：4rank×60层norm→idxKcopy launch之间cudaStreamWaitEvent全部0，launch提前约528ms；copy启动通常与restore结束对齐。restore SM资源竞争/调度是证据最支持的解释，尚未做受控资源实验。详细数据见trace_restore_prepare_findings.md。

Restore grid[35520,8,6]共1704960CTA。六个plane字节大小32768/32768/4096/4096/8192/1024，在tile4096下每page实际只需8+8+1+1+2+1=21tiles；grid每page48CTA，27个CTA会在tile*TILE<size判断处退出，959040CTA无payload复制。可精确flatten活跃plane/tile映射，去除56.25%空CTA；收益需验证，不能按CTA减少比例直接推算时间。

Norm grid7920×73共578160单warpCTA，GROUP8逐row静态循环；quant writer253440×9共2280960单warpCTA。Trace norm62regs/thread、quant42regs/thread，est occupancy均50%；这是profiler估计值，不是实测SM占用/瓶颈证明。可减少重复mapping、扩大row tile，保持group amax、stored E4M3 scale和E2M1 ties-to-even。

Compact prepare稳态GPU kernels18.5–18.9ms/rank，间隙3.1–3.3ms；launch中位提前约517.5ms。因此本次不是CPU未提交导致的稳态hostbound，但每层重复claim/remap/page-map清零可以按forward/chunk producer epoch预建，减少CPU enqueue和GPU kernel。IndexK值必须每层stage，OnlyScore fill删除需完整visible-write覆盖证明。

FI converter是主KV4布局转换，OnlyScore stage是独立indexK4→K8表示变换；两者不能通过共享同一张量消除。可以仅改变FI分支的主working destination，让prefix restore和suffix quant writer直接写FI planar（含K/V scale排列），persistent和index ABI保持原样；避免第三份主KV输出，才有望同时删converter和重复pool。

## 逐项实施记录（2026-10-09）

实施源树：`/data0/ruixuan.zrx/rtp_llm_m31_dspark_worktree`，起始 HEAD `2bd6b2a507be16559a574425e57f0229874a04b4`。已有 scheduler/server/distributed WIP 保留。历史 trace 使用冻结 package，不能视为新源码性能证据。实验输出：`/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_optimization_sequence`。

|顺序|用户问题与实施方向|正确性约束与验证|状态|
|---|---|---|---|
|1|Norm 直接生成连续 BF16 Q、E4M3 index Q 和 `[K,V,indexK]` suffix send；移除 rows copy、cat、index K contiguous 及逐 chunk Q cast。再评估 Norm row/head tiling。|保持原 reduction、partial NeoX RoPE、BF16 rounding 后 FP8 cast；V 原样复制；新模式不写原 projection，decode/旧 API 不变；字节一致、奇数尾行、高 position、独立 oracle、真实63360 rows CUPTI。|已接入；算子通过，模型验证待做|
|2|恢复 kernel flatten 活跃 plane/tile，消除56.25%空 CTA。|六 plane 字节完全一致；重复 source、唯一 dst、容量切片、越界、int64、Graph replay；35520页 matched benchmark。|同层调用已接入；算子通过，模型验证待做|
|3|prefix restore 与最后 suffix AG 的调度。|区分 AG-ready 和 restore-ready；评估重 restore 延后、独立计算流；固定 collective 顺序和 registered slot 生命周期，只做同层；CP4 timeline 证明收益。|待前两项验证|
|4|quant writer 每 CTA 多 row/plane，复用 row mapping。|working/persistent owner 字节一致；E4M3 stored scale、E2M1 ties-to-even、unpad/非owner/tail；253440 rows性能；避免盲目扩大寄存器。|CP调用已接入8rows；算子通过，模型验证待做|
|5|index prepare 的 page metadata 按 forward/chunk 缓存。|只有 immutable producer marker 生效时复用；direct API/Graph 修改 table 保留重建；page-list/remap每forward建立，K stage仍每层；计入512MiB预算。当前trace不支持host-bound结论。|已接入；CPU、GPU staging、真实OnlyScore/TopK通过，模型待做|
|6|OnlyScore score fill 的删除或融合。|先证明 native 覆盖所有 visible 分数；copy 读取 mask 包含 visible；边界、padding、causal、旧数据污染测试。没有证明就保留fill。|待覆盖验证|
|7|writer/restore 直接输出 FI working main layout，删除 `_convert_page_head` 和重复 main pool。|persistent/index ABI不变；FI四view共base/scale排列正确；禁止保留第三份全量working；partial-page tail清理保留；单算子+CP prefix+FI结果及内存验证。OnlyScore index K8 与 FI main KV4 不共张量。|待实施|

每阶段按单算子正确性 → 匹配性能 → 当前业务路径 timeline → PD4+4 质量、性能、驻留/峰值内存与 KV 容量推进。算子耗时不能直接相加成端到端收益。在线 package 不覆盖；完整服务切换前另记录源树、overlay、native so 和 workload 身份。阶段记录必须明确通过、失败和未测项。

### 第一批算子验证

运行环境：容器 `9c71264d3b58`，Torch `2.11.0+cu130`，Triton `3.6.0`，GPU0；服务安装目录保持原样。pytest依赖仅安装到实验目录的 `test_deps`，没有覆盖运行环境。初次pytest自动导入源码父package遇到缺少 `rtp_kernel`；改用直接加载单算子测试，避免把导入失败误报为数值失败。

- producer：rows 0/1/7/8/9/2049/63360、四种输入幅度、变更positions、alias拒绝和输出复用均通过；三个输出与 legacy norm+pack 逐字节一致。独立常量head/quarter-turn oracle、position81921、Graph多次回放通过。旧 indexK view及时释放，避免它继续持有整个projection storage。
- restore：原rectangular与active mapping共10项GPU测试通过，包括六plane独立容量/offset、非法索引、重复source、FP8 scales、Graph动态数据及非整4096 tile。same-layer调用启用active mapping；API默认旧路径仍可用于回归对照。
- CUPTI：10次交替、单stream、63360rows的legacy链路（原Norm+实际 `_rows_to_contig_kernel`+indexQ cast+cat）median kernel sum **1563.040us → 1076.731us**，降低31.11%。融合Norm自身约1077us，高于旧Norm约997us；收益来自整体materialization减少，不能宣称Norm本体已提速。
- CUPTI：35520prefix pages、37560destination容量、六plane真实尺寸，原restore **1380.944us → 868.458us**，降低37.11%。这是单算子无projection竞争的结果。
- index immutable cache：20项CPU测试通过，包含mutable table/epoch invalidation及预算；GPU staging/score和业务timeline尚待验证。

补充 index GPU 结果：`index_prepare_gpu.log` 记录3项测试及4个subtests通过。实际workspace构造、claim/remap、两chunk两层staging，与mutable重建和全poolstage逐字节一致；改变packed/scales后读出随层更新，safe-table缓存地址和内容保持。现有mutable Graph改table及独立reader参考也通过。OnlyScore planner/launcher在新workspace测试中mock，真实OnlyScore结果与端到端TopK尚未验证。

随后补齐真实OnlyScore对照：`index_prepare_real_onlyscore.log`记录1项通过（56.75s含首次JIT）。真实依赖为 `/data0/ruixuan.zrx/minimax-m31-dev/1007_prefill_1800/causal_prefix_fresh_candidate_dependency_v1/fmha_sm100/api.py`；CUTLASS从现有user site-packages加载，无安装修改、无planner/OnlyScore mock。64pages、两个4096row chunks、prefix4096/2048、重复页和invalid sentinel，跨两层改变packed/scales：immutable与mutable分数逐bit相等，生产canonical TopK IDs相等，invalid输出为-inf；第二层scores变化，cached safe-table身份和内容不变。此项补齐算子真实score/TopK对照，仍不是完整模型验证。

第一批源码检查：`git diff --check`与语法检查通过；同层workspace的16项CPU生命周期测试通过。原有scheduler/server/distributed改动未纳入本批patch。`source_manifest.json`记录本批文件hash，`source_changes.patch`提供限定范围diff。剩余stream调度、OnlyScore fill覆盖证明和FI working布局尚未实施；须先对本批进行匹配CP4模型timeline，避免把多项未经验证的变化叠加后失去归因。

Norm tiling sweep：GROUP 1/2/4/8/16 × warps 1/4/8共15种配置均逐字节一致。8/1仍最快，CUPTI mean1072.568us、末尾复测1074.300us；16/1=1098.030us，4/1=1127.611us。多warp需要512B shared并明显变慢，所有配置spill=0。保持原8/1，不把无收益的配置变更合入；`norm_tiling/results.json`保留反例和register/shared记录。

量化 writer：独立multirow kernel保留原1row接口默认；native CP调用选择8rows/CTA、4warps。R4/R8共12项GPU测试通过，涵盖1/3/4/7/8/9/127/128/129/1934尾行、四rank mapped reference、working/persistent独立有效性、无效source禁止读、非空Graph多次回放和int64 source溢出。零行Graph产生empty warning，正行Graph验证另行执行，不能用空Graph代替工作验证。

253440rows两次独立、各30轮交替CUPTI：原1row median1172.458/1173.898us；R8 median908.760/909.079us，改善22.5%；R4 median1380.844/1381.419us，退化17.8%，不启用。资源分别：原42regs/shared0，R8 47regs/shared512B，均spill0。benchmark为合成反序unpad和约25%owner，packed `[253445,1152]`；刻意一个无效row，实际working有效253439、persistent有效63359，不是生产BS60缓存map的逐项复现。计时前后working/persistent全部字节及未写哨兵均一致。证据归档到 `quant_writer/`；不将22.5%表述成端到端加速。

证据：`norm_correctness.log`、`norm_extra.log`、`restore_correctness.log`、`stage12_summary.json`、`norm63360.json`、`restore35520.json`。这些数字来自合成operator实验，未测端到端收益、生产精度或新分支的峰值内存；不能替代匹配BS60/80K/reuse75776 CP4模型timeline。

## 模型接入与后续并行批次

2026-10-09 用户要求第一批接入模型，并允许剩余批次同步推进。新campaign：`/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration`；源码HEAD已推进到 `399db00d611eddf52263a5ca3e2c09fae6ef01de`，第一批六个runtime文件独立冻结在 `stage1_source/`。

在线基线已变为 `1009_remove_attention_vendor/candidate` 的公共依赖版本。六个文件逐个核对：四个HEAD与installed字节相等，native score/TopK只有格式差异；不把更换provider收益算入本批。baseline先复用健康闲置服务重测；stage1只替换P的六个Python文件，D package和全部native so不变；不在活跃package目录覆盖文件。保持P CP4/EP4、D DP4/EP4、P/D KV预算65536/98304MiB、main/index KV4、同层ZeroCTA、next-layer prefetch OFF、D CG ON和normal rejection。

验收矩阵：BS40/60 80000 IDs、reuse75776，3轮无profiler；BS40→60→40容量增长/复用；自然PD40/60输出32 tokens；最后target的四rankCP4 timeline；输入hash和首token/输出token比较；新候选GSM8K500及LongBench160、native MAL；0.5s物理显存采样和ready/peak/growth。基线质量来自相同package及同ID的已完成provider campaign，明确区分其历史质量与本轮新测性能；必要时补充新生命周期质量A/B，不称为两臂都新跑。

后续三项各自基于第一批冻结副本开发，拥有独立worktree：`worktree_schedule`、`worktree_score_fill`、`worktree_fi_layout`。可并行源码推导、实现和CPU测试；GPU单算子按父任务时隙串行，模型包不合并未经验证的候选。模型对照按“第一批 → 加单个候选 → 再决定是否合并”执行，保持可归因。流调度和FI layout都涉及msa/writer/restore共享契约，最后由父任务协调合并，不让两位开发同时改同一份文件。


并行单算子 GPU gate（真实容器 Torch2.11/CUDA13，模型加载前串行执行，均已释放 CUDA context）：

- score-fill：CPU14项；GPU5项及10个subtest通过，真实OnlyScore、NaN旧scratch、ragged/128-page padding、CP padded tail、小query GQA、compact/full和Graph重放，score与TopK逐字节一致。必须同时让copy kernel只读取visible位置；不能盲删fill。192MiB scratch CUPTI kernel sum803.09→775.49us，CUDA-event833.92→831.26us，host launch间隙使单算子整体收益很小；workspace容量不变。尚未模型验证。
- FI direct working layout：CPU布局/契约和8个离线SM100编译通过；GPU真实prefix/suffix字节oracle、persistent/index不变、partial tail及两段FI输出exact、37560-page超过2GiB池末页验证通过。73728B/page共同storage的非连续view直接消费；理论删除重复池2769223680B（2.579GiB），不是已测allocator峰值。小fixture事件时间0.35808→0.31242ms，不能当生产性能。尚未模型验证。
- schedule：CPU21项，真实CP4 CE/ordinary small及large字节/lifetime均通过，ZeroCTA trace没有NCCL CTA。large prefix8880/suffix63360 restore约888us，但四rank均没有PtoP/restore重叠，suffix PtoP在restore结束约51us后开始；当前不得以有效掩盖优化推广。继续定位NCCL内部stream依赖。测试25.29GiB峰值包含oracle/pattern临时，不能当生产workspace开销。

基线profile请求API返回成功但未落盘：原在线服务的TORCH输出base未创建，含`../..`路径不能绕过不存在的中间目录；原服务已退役，trace不能追溯恢复。第一次无profiler正式性能有效；必须新增baseline_repeat冷启动控制和四rankprofile再完成timeline比较。stage1输出base已提前创建。基线原服务已有历史allocator高水位，不能把新stage1较低ready显存直接归因优化；baseline_repeat用于同生命周期内存控制。


第一批实际模型无profiler结果（2026-10-09，本轮基线同public provider，三轮warm中位数）：BS40 1.792699→1.759195s（1.87%），BS60 2.677169→2.570943s（3.97%），BS40 grow/shrink 1.799475→1.759452s（2.22%）；自然PD40 1.992542→1.956190s（1.82%），PD60 2.877883→2.796597s（2.82%）。自然PD两次context forward的native sum，不能称作单次forced forward。全部请求成功，质量仍单独核验。

候选四rank实际trace已落盘。最后target rank0 2559.459ms；rows-copy/index-Q-cast 0，claim/remap各4，restore60、grid745920，R8 writer60、grid31680×9，norm60总68.345ms；index stage240总9.642ms、fill240总5.617ms、OnlyScore240总245.181ms、TopK240总135.732ms、converter60总48.976ms、FI attention60总432.789ms。8个metadata CatArray总0.060ms仍在，不能声称所有cat消失。匹配baseline_repeat trace后再给每项实际A/B及通信掩盖。

合成ID负载的输出并非逐条完全相同：同臂重复也有变化（例如baseline BS60 warm1对warm2/3均50/60完全相同，stage1为47/60、50/60，A/B warm1为48/60）。不能据此单独归因某算子精度；输入hash逐项一致，继续同ID质量和operator逐字节验证。


### 第一批本轮新基线控制（模型验证已完成）

Baseline_repeat重新冷启动 unchanged public-provider 包，P889/D888个.py/.so逐文件hash一致；BS40→60→40各三轮warm、实际四rank非空trace及660条新质量均完成。三组native model中位数：1.807021→1.759195s（2.65%）、2.654948→2.570943s（3.16%）、1.818289→1.759452s（3.24%）。这组控制优先用于当前纯prefill结论；前一次自然PD性能仍有效，未重复自然PD控制。

同ID/逐条request相同的GSM8K均435/500=87.0%；官方LongBench raw66.603217→66.614550，均5条truncation；native MAL分别GSM3.997827→3.958954、LongBench3.195164→3.170358。GSM298/500、LongBench158/160完整response相同；不能据此声称内部全trajectory位相同或某算子因果精度变化。operator层的逐字节和真实OnlyScore/TopK一致性证据另见单算子gate。

最后target rank0实际profile2645.545→2559.459ms、kernel5695→4271。累计GPU duration(ms)：norm63.154→68.345（本体慢5.191），rows-copy18.977→0，KV/indexK cat11.434→0.060（68→8次，剩metadata），restore85.187→54.019（少31.168），writer73.397→57.355（少16.042），claim0.754→0.011（240→4）、remap0.630→0.007（240→4）、indexQ-cast1.482→0（240→0）。stage-index/OnlyScore/TopK/converter/FI主体基本不变。不能将GPU duration sums简单相加当CPU关键路径收益。Norm单独未更快，收益来自producer整链及restore/writer。

第一批0.5s物理显存采样没有证实所有P卡整体peak降低：BS60 warm-window baseline_repeat四卡201349/178927/179709/179705MiB，stage1为200849/180921/179271/180019MiB；allocator/高水位有卡间差异。采样仅perf+profile阶段，per-case峰值是warm正式RPC窗口，不覆盖load/capture/quality，不能叫全生命周期峰值，也未改变KV预算或cache容量。

调度第二gate：corrected direct和async内部NCCL流均四rank字节/lifetime通过，但PtoP仍无restore重叠。已采sync events，通信流未显示restore-ready消费wait；缺CE memory-op关联，无法精确归因跨rankbarrier还是driver调度，不声称硬件必然限制。未合入此候选。


### FI direct初版：模型暴露退化，已定位并修正候选

FI初版完整模型五case/四ranktrace/660质量已完成，GSM435/500，官方LongBench66.875794（5条truncation）。相对stage1 pureBS40 1.759195→1.782584s（慢1.33%），BS60 2.570943→2.620280s（慢1.92%）。不以删除converter宣称模型加速。采样BS60 warm窗口P四卡peak由200849/180921/179271/180019降至198829/177085/177789/178833MiB；实际物理减少1.16–3.75GiB，包含allocator差异，不是严格等于理论2.579GiB，也未增加KV budget/capacity。

matched最后target rank0：restore54.019→138.076ms（60次、128regs）；writer57.355→77.605ms（47→65regs）；converter48.976→0；FI主体432.789→421.616ms。CPU target2559.459→2585.208ms，不能把全部GPU duration delta相加当关键路径。

SM103 PTX/cubin证实restore具体源码低效：runtime scale plane分支的source_x/output_x PHI使main/index flatcopy丢失连续性，原ld/st.global.v4.b32退化为32条b8，active regs20→128；不是仅凭模型波动猜原因。独立修正把scale gather/store留在自己分支，其余plane保留原x连续load/store；V scale按physical output order反向查MMA来源，目的x线性。修正非FI仍20regs，FI active48、rect40；packed writer无同类vector-width退化，暂未改它。

fixed_restore SHA256425c1a0ef7e7bced8f294e5e631b222124f9e1c0506687498d7b2f6f8031319b。真实GPU第三gate：35520 restore IDs、37560 pool>2GiB、active/rect两launch、六plane/sentinel exact、NaN opaque字节/invalidskip、Graph每launch三revision、FI尾页输出old/fixed exact finite及独立FP32 oracle（RMS相对误差0.002069）均通过。10轮matched event active2.208864→1.085184ms（50.87%），rect2.501296→1.654752ms（33.84%）；CUPTI确认active128→48regs，原grid745920保持。初版运行包及证据未覆盖；新fi_restore_stage1四文件包独立冻结，接在score-fill模型之后验证。

Suffix先量化再AG的后续设计仅在schedule_candidate/wire_nvfp4_design.md：global scale=1且16-channel row-local，保留BF16 rounding和原算术，2304B/row→648B/row理论减少71.875%；需要新的wire/register lease ABI和bit-exact工作/persistent scatter验证。尚未实现/测试，不能算已取得收益。


## 后续 batch3 实施边界

用户已授权继续后续优化和验证。模型 critical path 为 score_recheck_stage1 → fi_restore_stage1；两者各自运行固定五场景、四rank真实trace、GSM8K500和LongBench160及perf/profile显存采样。之后才允许GPU单算子任务。所有已有失败和无效复用补测保留，不参与性能A/B。

并行源码候选独立隔离：

- norm/RoPE：以冻结stage1为基线，保留Q/indexQ/K的BF16 rounding、FP8 scale和registered suffix载体；优先排查CTA、向量访问及寄存器压力。
- index compact staging：以现有score-fill候选为基线，保持claim/remap/sentinel/CP padding语义；不得增加CPU同步或跨层缓存旧indexK。
- suffix NVFP4 wire：以冻结stage1为基线，sender按原算术量化，receiver按原maps散布六planes；只在实验选择下启用。理论648B/row和71.875%传输减少尚非实测性能结论。

CPU/离线编译通过后，必须追加实际GPU字节/精度/Graph/大stride及matched timeline验证；通过后再接模型。最终组合未经单独模型验证不能宣称各单变量收益可相加。主树已有其他Decode/C++ WIP，所有后续patch只按拥有的路径增量集成。


### Batch3 已完成的负收益 gate

- Norm gamma hoist、GROUP16：实际SM103 50项正确性/Graph/大stride通过，63360 event中位old1118.576us、hoist1118.208us（仅0.033%）、GROUP161148.528us（更慢）。没有推广。
- Norm追加GROUP4、warp2、warp4：数学AST和实际输出bitexact，63360 old1118.768us，分别1165.248/2233.648/3158.848us，均更慢；42240亦检查，保留GROUP8/warp1。warp2/4增加1024B shared和barrier且缩窄global访问。
- Index scale-view、compact stage warp8：156个真实case、22份CUPTI通过，Actual OnlyScore/TopK/eager/Graph/动态maps/内容/padding/sentinel检查均过。warp8 stage慢0.27–3.61%，八组integration event全部慢1.11–2.11%；view七/八组event更慢，无稳定收益，均不合入。
- 单算子测试时FI模型已完成全部请求并idle，但保留模型显存；不称empty GPU，不将GPU event或静态编译收益直接当模型收益。

对应证据：norm_batch3_gpu_gate、index_prepare_gpu_gate，均位于当前campaign目录。suffix wire真实CP4链路测试进行中，未接入模型。


### suffix wire v1 实测与迭代

648B/row初版单GPU67项检查通过（MMA/FI working两布局、persistent MMA、Graph内容/maps变化、>2GiB、isolated invalid assertion）。CP4 ordinary/CE small queued及大规模63360 local→253440 full receiver、反序rank-aligned各四rank字节均通过。实际关键链性能失败：CE aligned旧约5.39–6.38ms，wire约9.67–12.07ms；旧writer约0.92ms，初版scatter约5.18–7.68ms。one-row/plane scatter产生过多CTA，不能把payload71.875%减少当作速度提升。最初old-first queued arrival-skew计时已排除性能结论，原证据仍保留。

默认OFF候选未合入主树；正在保持648B wire/sender/ABI不变，独立测试multirow receiver scatter，先CPU/离线，再实际GPU字节和matched单算子，通过后才重复CP4链路/接模型。

FI writer初fixture persistent planar stride与生产native65536/17408 ABI不同，首轮R1收益不得用于模型建议；正在以cache_layout.logical_views重跑真实ABI。


### FI writer 真实ABI与wire R16待模型候选

FI writer native65536/17408 persistent strides的R8 TTIR与实际模型cache只规范化源码路径后逐字一致，寄存器65，排除初planar fixture的74regs。真实253440行R8→R1 event1308.528→1253.104us（4.24%），CUPTI1246.374→1200.040us（3.71%），10/10轮赢；R4慢13.47%。byte/Graph/>2GiB通过，已准备FI-only一行模型arm fi_writer_r1_stage1，non-FI保留R8，尚未运行/合入。

wire R16修正版保持648B ABI/原sender/maps/layout/persistent，只减少receiver CTA：full253440行grid2280960→142560；MMA/FI各六working/persistent、Graph、大stride通过。R16 scatter event约0.307456ms，local63360 sender约0.31768ms，连续sender+fullreceiver约0.548464ms；这些未含AG、不代表模型收益。真实FI CP4 ordinary/CE及prefix8880完整链验收进行中。

只读审查：FI已集成四文件无阻塞finding，主树hash吻合；wire旧candidate_delta.patch/hash落后于已测试空Tensoralias修正/FI支持，需生成当前R16完整patch及manifest并保持旧版证据。FI组合MSA调用必须显式传fi_working_layout selector，默认False的单独wire树不能直接覆盖FI主树。

### 2026-10-10 最终 CP4 验收与吞吐交付约定

本轮最终模型 arm 为 `final_wire_cp4`：冻结已通过完整模型验证的
`fi_restore_stage1`，仅组合 R16 suffix NVFP4 wire，P 端
`RTP_LLM_CP_SUFFIX_NVFP4_WIRE=1`、`NCCL_CTA_POLICY=2`，D 端保持相同包。
R1 writer、norm/index launch、score-fill 和新 overlap schedule 本轮后置。
当前状态：组合源码与 CPU 审核准备中，尚无该 arm 的模型性能或质量结论。

完成验收后按原 TP4/EP4/CP4 配置执行正式 native token-ID perf，覆盖 pure
BS40/60、BS60 后缩回 BS40，以及 natural PD BS40/60；每项一轮 cold、三轮
warm，另采集 BS60 四 rank timeline。保持 input=80000、reuse=75776，所有
request 验证实际复用与 native context_batch/execute_tokens。

单卡带 cache TPM = `BS * 80000 / 4 / median(warm model_forward_s) * 60`；
TPMS = 同样的每卡总输入 token 数除以模型毫秒数。使用
`NormalExecutor.model_forward_us`，natural PD 多 context step 累加，不使用
客户端 wall、profile duration 或 aux cost_time 代替；95% 为近似称呼，实际
命中比例为 94.72%，每请求实际计算 4224 token。

最终结果由 `1009_prefill_model_integration/final_cache_throughput.py` 从正式
terminal 重新核对生成，记录三轮原始 model latency、context step 数、
TPM/TPMS。完成后再更新此处的状态与 evidence 链。


### 2026-10-10 最终 CP4 性能与质量采集完成


性能测试与质量采集完成，wire 的生产精度验收尚未签收。以下为 `final_wire_cp4`、suffix wire=1 的实测结果；main 实验开关默认 OFF，已验证 FI 路径保留。

P TP4/EP4/CP4、KV sharded、Zero-CTA、同层路径、next-layer prefetch OFF、P CG OFF；D TP1/DP4/EP4、CG ON、正常 DSpark rejection。固定 checkpoint `MiniMax-M3.1-preview2-dspark`，native 库与 D 包未改，七文件 Python overlay 冻结。

960/960 正式请求成功，五场景各 cold1+warm3，逐请求 input=80000/reuse=75776（94.72%），每请求实际计算4224 IDs；native context batch 与 execute_tokens 全部核对。Pure为1个context step，natural PD为2步累加。

| 场景 | FI 模型秒 | wire 模型秒 | 耗时下降 | 单卡 cache TPM | 单卡 cache TPMS |
|---|---:|---:|---:|---:|---:|
| Pure BS40 | 1.734955 | 1.692255 | 2.461% | 28,364,520 | 472.742 |
| Pure BS60 | 2.552296 | 2.489655 | 2.454% | 28,919,670 | 481.994 |
| Pure BS60→40 | 1.739498 | 1.719589 | 1.145% | 27,913,647 | 465.227 |
| Natural PD BS40 | 1.924397 | 1.894663 | 1.545% | 25,334,321 | 422.239 |
| Natural PD BS60 | 2.779305 | 2.723603 | 2.004% | 26,435,571 | 440.593 |

TPM=tokens/minute，TPMS=tokens/millisecond。分子包含 cache 命中的全部输入 IDs，均摊4张P卡；分母为三轮warm `NormalExecutor.model_forward_us` 的中位数。不是client wall吞吐、不是live KMonitor，profile不进入正式计时。原始三轮见 cache_throughput.json；RPC完整响应和TTFT另见性能报告。

相对本轮 fresh unchanged baseline_repeat，Pure BS40/60 从1.807021/2.654948秒降至1.692255/2.489655秒，耗时下降6.351%/6.226%。baseline_repeat 只有三个pure场景，不借用其值计算natural PD基线。

## 实际四 rank timeline

每rank最后一次target forward：rank0–3为2472.886/2472.943/2473.003/2473.024ms。每rank均60次sender、60次R16 scatter；旧suffix writer为0，converter为0。rank0 sender18.470ms+scatter19.108ms替代旧FI writer78.110ms；norm68.008ms（FI67.890ms）、restore64.317ms（FI64.046ms）。kernel数4211→4271：增加pack，不代表变慢。

每层suffix local copy 145,981,440→41,057,280 bytes，peer copy437,944,320→123,171,840 bytes，减少71.875%。prefix两AG不变。rank0 suffix peer copy共10.086ms。四rank suffix peer-copy与restore交叠均0，生产仍同SIDE stream FIFO；这是减传输/写入成本，不是新的AG overlap。duration sum不能跨stream相加当target latency。

## 质量与内存边界

同660请求IDs、input与官方scorer：GSM8K 436/500→436/500，MAL3.95936→4.01421；LongBench raw66.8171467→66.2377381，MAL3.18117→3.22242。LB156/160输出相同，4条变化；两个arm都5条截断且IDs相同。主要降分是一题qasper由正确yes变为no（100→0，对全体raw贡献−0.625）；差异尚未做首token logits/最早不同内部状态定位，不能宣称因果回归已排除或精度等价。GSM全文298/500相同，也不以总分相同推断逐token一致。

BS60 warm P卡0–3物理抽样峰值（MiB）：FI [199639,177241,177631,178285]；wire [199381,176615,177425,177615]。0.5s抽样覆盖perf+profile，排除load/capture/quality，不是完整生命周期峰值；固定KV预算没有增加，不宣称KV容量提升。

## 交付及后续

稳定第一批与FI路径保留；wire R16已以默认OFF实验开关增量接入主树。七runtime文件与model snapshot相同，MSA仅保留主树两处既有Decode阈值160；其他Decode/C++ WIP逐文件hash未变。主树34 wire CPU+20 workspace测试、语法与diff check通过。测试fixture更新为已验证FI fallback并检查显式FI selector；旧stage1 fixture失败、host缺numpy和报告FileExists均保留，均不是GPU/model请求失败。原controller因提前生成的汇总报告禁止覆盖退出，预览报告移为prequality，最终报告重新生成；未重跑/篡改正式模型数据。

没有稳定收益的norm/index/score-fill、R1 writer、direct/async调度后置。AG新overlap仍未完成；wire精度复核需固定单个yes/no ID，先观察首token logits及最早不同内部状态，不扩展新矩阵。

## 证据

- [单卡 cache 数值与三轮原始耗时](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/cache_throughput.json)、[CSV](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/cache_throughput.csv)
- [wire字节与四rank交叠审核](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/wire_timeline_audit.json)
- [模型完成记录](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/terminal.json)、[源码/native hash检查](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/launch_hash_preflight.json)
- [FI对照性能](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/performance_fi_restore_stage1_final_wire_cp4.json)、[fresh baseline对照](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/performance_baseline_repeat_final_wire_cp4.json)
- [FI对照质量](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4_quality_comparison_fi_restore_stage1.json)、[实验接入receipt](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/wire_experimental_integration/receipt.json)
- [四 rank timeline prefill_wr0_1.json](/data0/ruixuan.zrx/profiling_results/m31_prefill_stage1_model_ab_20261009/final_wire_cp4/bs60/prefill_wr0_1.json)
- [四 rank timeline prefill_wr1_1.json](/data0/ruixuan.zrx/profiling_results/m31_prefill_stage1_model_ab_20261009/final_wire_cp4/bs60/prefill_wr1_1.json)
- [四 rank timeline prefill_wr2_1.json](/data0/ruixuan.zrx/profiling_results/m31_prefill_stage1_model_ab_20261009/final_wire_cp4/bs60/prefill_wr2_1.json)
- [四 rank timeline prefill_wr3_1.json](/data0/ruixuan.zrx/profiling_results/m31_prefill_stage1_model_ab_20261009/final_wire_cp4/bs60/prefill_wr3_1.json)

用户最新优先级：LongBench这一轮分差先记录，性能优先，本轮不再展开精度复测矩阵。最终完整报告：[/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/RESULT.md](/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/RESULT.md)。模型验证容器当前五endpoint alive、running/waiting=0，暂保留空闲驻留，后续Decode验证可受控切换。
