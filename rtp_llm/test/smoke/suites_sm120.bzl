load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name = "softmax_probs_sm120",
                task_info = "data/model/qwen25/q_r_softmax_probs_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "fp16_sm120",
                task_info = "data/model/qwen25/q_r_s_fp16_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "bf16_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "bf16_cuda_graph_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0 --seq_size_per_block 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "random_seed_sm120",
                task_info = "data/model/qwen25/test_random_seed_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "logits_index_sm120",
                task_info = "data/model/qwen25/logits_index_q_r_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    native.test_suite(
        name = "smoke_sm120_dense",
        tests = [
            smoke_test(
                name = "dense_fp8pb_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8pb_dynamic_cudagraph_sm120",
                task_info = "data/model/qwen3/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0 --enable_cuda_graph 1",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "dense_fp8pt_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            # TODO(PR-4 followup): frontend_app_sm120
            #   blocker B-6: smoke_args dict 同时声明 frontend + pd_fusion
            #   两个 role，gpu_count=2，本机单卡 RTX 5000 Pro 起不来。
            #   CI sm12x stage 是 2 卡 → 这个 case 需要去 CI 实测一次再落 case。
            # TODO(PR-4 followup): tp2_sm120 / beam_search_tp2_sm120
            #   单卡 RTX 5000 Pro 跑不了；需要 2 卡环境（无 NVLink，PCIe TP）
            # Qwen3-1.7B greedy (top_k=1, max_new_tokens<=10), 模型已下载到
            #   /mnt/nas1/hf/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5
            # Qwen3-1.7B 没有 L20 baseline，golden 由 sm120 自录。
            smoke_test(
                name = "dense_fp8kv_cudagraph_sm120",
                task_info = "data/model/qwen25/q_r_fp8_kv_cache_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 1",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen3_1_7b_prequant_fp8pb_sm120",
                task_info = "data/model/qwen3/q_r_1_7b_prequant_fp8_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--act_type BF16 --warm_up 0 --enable_cuda_graph 1",
                gpu_type = ["RTX_5000_PRO"],
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
        ],
    )

    # RTX 5000 Pro has no NVLink, so use the PureTP router and the SM120
    # Triton grouped-GEMM executor rather than DeepEP.
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            smoke_test(
                name = "moe_fp8pb_tp2_sm120",
                task_info = "data/model/qwen3_moe/q_r_30b_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
                gpu_type = ["RTX_5000_PRO"],
            ),
            # ========== PR-5 扩展 (Tier 1: BF16 + dynamic FP8 across Qwen2.5-0.5B / Qwen3-8B) ==========
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
        ],
    )
