load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    # SM120 / RTX 5000 Pro / Blackwell consumer (GB202)
    # ============================================================================
    # 硬件约束（写死规则，决定哪些 case 可以放进来）：
    #   - 单卡 32GB VRAM；TP=2 走 PCIe，DeepEP-LL 因无 NVLink 不可用
    #   - sm_120a CUDA arch；FlashInfer v0.6.12rc1+rtp.260523 已带 b12x AOT
    #   - cuda12_9 工具链
    # 维护规则：
    #   - 每个 PR-N 跑通才把对应 case 落到这里（避免 CI 噪声）
    #   - 命名 `<case>_sm120` 后缀，避免和 sm8x/h20 同名 py_test 冲突
    #   - mainse_bert 等内源协议 case 放在 internal_source/.../suites_sm120_internal.bzl
    # ============================================================================

    # SM120 Basic (sm8x_basic 同位 — Qwen2.5 dense FP16/BF16)
    # 对应 PR-4: Qwen dense FP16/BF16
    # 跨架构 golden 注意:
    #   - softmax_probs / fp16 / bf16 用 max_new_tokens<=10 + top_k=1 greedy，
    #     golden 在 L20/H20/SM_120 上稳定，可以直接复用 L20 task_info
    #   - random_seed (top_k=100 采样) / frontend_app (frontend+pd_fusion 双角色)
    #     在 sm_120 上各自踩到不同 blocker，参见下方 TODO 注释
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name="softmax_probs_sm120",
                # sm120-specific golden: FMHA on sm_120a uses a different code
                # path than L20, producing ~1e-3 numerical drift on softmax
                # probabilities. fp16/bf16 above use top_k=1 greedy and stay
                # byte-identical to L20 golden, so they share q_r_s.json.
                task_info="data/model/qwen25/q_r_softmax_probs_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="fp16_sm120",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="bf16_sm120",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # Qwen2.5 sampler path: top_k=100 + random_seed=46. sm_120a 上采样
            # 结果与 L20 不 bit-equivalent（不同 FMHA → 不同 softmax probs →
            # 不同 sampled token），所以独立录 sm120 golden。
            smoke_test(
                name="random_seed_sm120",
                task_info="data/model/qwen25/test_random_seed_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # Qwen2.5 OpenAI route + return_logits + select_tokens_id + logits_index.
            # logits values are non-deterministic across GPUs (comparer 只比 shape)，
            # response 文本在 top_k=1 greedy 下可能与 L20 byte-identical，但
            # OpenAI chat 路径 cost_time / first_token_cost_time 字段每次不同，
            # 独立录 sm120 golden 保险。
            smoke_test(
                name="logits_index_sm120",
                task_info="data/model/qwen25/logits_index_q_r_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # TODO(PR-4 followup): frontend_app_sm120
            #   blocker B-6: smoke_args dict 同时声明 frontend + pd_fusion
            #   两个 role，gpu_count=2，本机单卡 RTX 5000 Pro 起不来。
            #   CI sm12x stage 是 2 卡 → 这个 case 需要去 CI 实测一次再落 case。
            # TODO(PR-4 followup): tp2_sm120 / beam_search_tp2_sm120
            #   单卡 RTX 5000 Pro 跑不了；需要 2 卡环境（无 NVLink，PCIe TP）
            # TODO(PR-4 followup): embedding_qwen_gte_7b_cudagraph_sm120
            #   依赖 gte-Qwen2-7B (~14GB) 模型下载 + CUDA Graph 独立验证
            # Qwen3-1.7B greedy (top_k=1, max_new_tokens<=10), 模型已下载到
            #   /mnt/nas1/hf/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5
            # task_info 是 placeholder result，第一次跑 sm120 前需 --config=rewrite_smoke
            # 录 golden 进 source tree（与 fp16_sm120/bf16_sm120 共享 q_r_s.json 不同：
            # Qwen3-1.7B 没有 L20 baseline，golden 由 sm120 自己生成）
            smoke_test(
                name="qwen3_1_7b_fp16_sm120",
                task_info="data/model/qwen3/q_r_qwen3_1_7b_greedy.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen3_1_7b_bf16_sm120",
                task_info="data/model/qwen3/q_r_qwen3_1_7b_greedy.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # 跨系列覆盖：Qwen3-8B dense (medium)，OpenAI chat completions 路径 +
            # max_new_tokens<=10 + top_k=1 greedy；BF16/FP16 byte-identical 共享 task_info。
            smoke_test(
                name="bf16_qwen3_sm120",
                task_info="data/model/qwen3/q_r_sm120.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="fp16_qwen3_sm120",
                task_info="data/model/qwen3/q_r_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # 跨系列覆盖：Qwen2 dense (0.5B)，OpenAI chat completions 路径 + greedy；
            # 跟 Qwen2.5 同 model_type=qwen_2，但模型权重不同，覆盖 Qwen2 系列。
            smoke_test(
                name="bf16_qwen2_sm120",
                task_info="data/model/qwen2/q_r_sm120.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="fp16_qwen2_sm120",
                task_info="data/model/qwen2/q_r_sm120.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # gte-Qwen2-7B-instruct DENSE_EMBEDDING + CUDA Graph capture/replay。
            # 覆盖 sm8x_basic 的 embedding_qwen_gte_7b_cudagraph 同位；CUDA Graph
            # 是 sm_120 上的新踩点（实测一次 PASS，无新 blocker）。
            # golden 用 sm120 自录的 gte-embedding_sm120.pt 浮点向量。
            smoke_test(
                name="embedding_qwen_gte_7b_cudagraph_sm120",
                task_info="data/model/qwen2/q_r_embedding_sm120.json",
                smoke_args="--seq_size_per_block 64 --embedding_model 1 --act_type BF16 --concurrency_limit 2 --enable_cuda_graph 1  --enable_cuda_graph_debug_mode 1 --prefill_capture_config '150,155,160,380,400' --task_type DENSE_EMBEDDING --reserver_runtime_mem_mb 3072",
                gpu_type=["RTX_5000_PRO"],
            ),
            # Qwen3.5 Dense 4B (hybrid: standard attention + GDN linear attention + causal conv1d)
            # 首次在 sm120 上验证 qwen35_dense model_type + FLA Triton kernels
            # --seq_size_per_block 2048: 增大 full-attn KV block 使 HybridConfigCreator
            #   通过 full_block >= linear_block 约束（同 H20 qwen35 测试参数）
            smoke_test(
                name="qwen35_dense_bf16_sm120",
                task_info="data/model/qwen35/qwen35_dense_bf16_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--act_type BF16 --seq_size_per_block 2048",
                gpu_type=["RTX_5000_PRO"],
            ),
        ],
    )


    # SM120 Dense (h20_dense 同位 — Qwen3 dense FP8)
    # 对应 PR-5: Qwen dense FP8 PER_BLOCK
    # 关键: envs=["LOAD_PYTHON_MODEL=1"] 强制走 Python LinearFactory →
    #   FP8_PER_BLOCK 路由到 CudaFp8VllmBlockwiseLinear
    #   (sm_120 上 DeepGEMM 无 cubin / FlashInfer PB stalls — 见 fp8_gemm_linear.py
    #    can_handle 的 is_sm12x() gate；这条 gate 与 --disable_flash_infer 无关)
    # 注意：sm120 不能加 --disable_flash_infer 1（PR-3 之后 FlashInfer 是 sm_120a
    #   唯一可用的 FMHA backend；trtllm/xqa 都在 sm120 上被 gate 关了）。
    # task_info: FP8 PER_BLOCK 量化噪声使 sm120 输出 token 与 h20 不 bit-equiv，
    #   独立录 q_r_fp8pb_sm120.json (329 tokens vs h20 570 tokens, 语义正确但分支不同)
    native.test_suite(
        name = "smoke_sm120_dense",
        tests = [
            smoke_test(
                name="dense_fp8pb_dynamic_sm120",
                task_info="data/model/qwen3/q_r_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="dense_fp8pt_dynamic_sm120",
                task_info="data/model/qwen3/q_r_h20_per_tensor_w13.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="dense_fp8kv_cudagraph_sm120",
                task_info="data/model/qwen25/q_r_fp8_kv_cache_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 1",
                gpu_type=["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 1: BF16 + dynamic FP8 across Qwen2-0.5B / Qwen2.5-0.5B / Qwen3-8B) ==========
            smoke_test(
                name="qwen2_0_5b_fp8pb_sm120",
                task_info="data/model/qwen2/q_r_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_0_5b_fp8pt_sm120",
                task_info="data/model/qwen2/q_r_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_5_0_5b_fp8pb_sm120",
                task_info="data/model/qwen25/q_r_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen2_5_0_5b_fp8pt_sm120",
                task_info="data/model/qwen25/q_r_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen3_8b_fp8pb_sm120",
                task_info="data/model/qwen3/q_r_8b_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="qwen3_8b_fp8pt_sm120",
                task_info="data/model/qwen3/q_r_8b_fp8pt_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 2: 预量化 FP8 model load 路径) ==========
            # Qwen3-1.7B-FP8: HF quant_method=fp8 + weight_block_size=[128,128] → CudaFp8VllmBlockwiseLinear (PB)
            smoke_test(
                name="qwen3_1_7b_prequant_fp8pb_sm120",
                task_info="data/model/qwen3/q_r_1_7b_prequant_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # Qwen2.5-7B-Instruct-FP8: HF quant_method=fp8 dynamic (无 weight_block_size) → CudaFp8PerTensorLinear
            smoke_test(
                name="qwen2_5_7b_prequant_fp8pt_sm120",
                task_info="data/model/qwen25/q_r_7b_prequant_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--act_type BF16",
                gpu_type=["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 3: FP8 路径上的边角 case，参考 PR-4 random_seed/logits_index 模板) ==========
            smoke_test(
                name="dense_fp8_random_seed_sm120",
                task_info="data/model/qwen3/test_random_seed_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            smoke_test(
                name="dense_fp8_logits_index_sm120",
                task_info="data/model/qwen3/logits_index_fp8_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
            # Qwen3.5 Dense 4B FP8_PER_BLOCK (hybrid attention + GDN + conv1d)
            # BF16 case 在 smoke_sm120_basic 中通过后追加 FP8 量化路径验证
            smoke_test(
                name="qwen35_dense_fp8pb_sm120",
                task_info="data/model/qwen35/qwen35_dense_fp8pb_sm120.json",
                envs=["LOAD_PYTHON_MODEL=1"],
                smoke_args="--quantization FP8_PER_BLOCK --act_type BF16 --seq_size_per_block 2048 --warm_up 0",
                gpu_type=["RTX_5000_PRO"],
            ),
        ],
    )


    # SM120 MoE (h20_moe + sm100_moe 同位 — Qwen3-30B MoE)
    # 对应 PR-6 (FP8) + PR-9 (NVFP4 可选)
    # 注意：RTX 5000 Pro 无 NVLink → DeepEP 路径全部不在此 suite 内
    #       仅跑 fp8_per_block_no_dp_masked / pure_dp / nvfp4 non-deepep
    # TODO(PR-6): moe_masked_fp8_sm120
    # TODO(PR-9): moe_nvfp4_no_deepep_sm120


    # SM120 Next (h20_next 同位 — Qwen3-Next BF16/FP8 + linear attention)
    # 对应 PR-7: Qwen3-Next + linear attention
    # TODO(PR-7): next_bf16_basic_sm120 / next_fp8_basic_sm120
