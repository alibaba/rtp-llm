load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            # Real qwen_2 model, TP2 NCCL INT8 communication, both deployment
            # switch values. Reuse the established BF16 output golden.
            smoke_test(
                name = "qwen2_int8_fp16_scale_tp2_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--act_type BF16 --tp_size 2 --world_size 2 --max_seq_len 4096 --test_block_num 256 --enable_cuda_graph 1 --decode_capture_config '1,2' --enable_int8_allreduce_fp16_scale true",
                gpu_type = ["RTX_5000_PRO_CU13", "RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_int8_bf16_scale_tp2_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--act_type BF16 --tp_size 2 --world_size 2 --max_seq_len 4096 --test_block_num 256 --enable_cuda_graph 1 --decode_capture_config '1,2' --enable_int8_allreduce_fp16_scale false",
                gpu_type = ["RTX_5000_PRO_CU13", "RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_mtp_int8_fp16_scale_tp2_sm120",
                task_info = "data/model/qwen2_14b/q_r_mtp.json",
                smoke_args = "--act_type BF16 --warm_up 0 --ft_disable_custom_ar 1 --tp_size 2 --world_size 2 --max_seq_len 4096 --test_block_num 256 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4,5' --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --enable_int8_allreduce_fp16_scale true",
                gpu_type = ["RTX_5000_PRO_CU13", "RTX_5000_PRO"],
            ),
            smoke_test(
                name = "qwen2_mtp_int8_bf16_scale_tp2_sm120",
                task_info = "data/model/qwen2_14b/q_r_mtp.json",
                smoke_args = "--act_type BF16 --warm_up 0 --ft_disable_custom_ar 1 --tp_size 2 --world_size 2 --max_seq_len 4096 --test_block_num 256 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4,5' --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --enable_int8_allreduce_fp16_scale false",
                gpu_type = ["RTX_5000_PRO_CU13", "RTX_5000_PRO"],
            ),
            # Qwen3.5 uses independent pools for full-attention KV pages and
            # recurrent linear-attention states. Reusing one 64-token block
            # exercises the first GDN state boundary on the SM120 Triton path.
            smoke_test(
                name = "qwen35_dense_bf16_block64_reuse_sm120",
                task_info = "data/model/qwen35/qwen35_dense_bf16_block64_reuse_sm120.json",
                smoke_args = "--act_type BF16 --seq_size_per_block 64 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 4096 --tp_size 1 --world_size 1 --reuse_cache 1",
                gpu_type = ["RTX_5000_PRO"],
            ),
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
            # Answer-mode no-think contract on the SM120 pool: think_mode=0 plus a
            # caller that declares neither response_format nor template kwargs, so
            # the endpoint aligns the template's thinking switch, compiles the
            # no-think envelope itself, and the only caller-visible contract is the
            # answer shape. Kept in this suite -- a member of maga_model_smoke_light
            # -- so the sm12x pool actually runs it.
            smoke_test(
                name = "qwen35_answer_mode_no_think_sm120",
                task_info = "data/model/qwen35/q_r_no_think_answer_mode_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0 --max_seq_len 4096 --tp_size 1 --world_size 1 --think_mode 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 dense FP8; FP8_PER_BLOCK is routed through the shared DeepGEMM path.
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
                name = "dense_fp8pt_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 MoE FP8 (RTX 5000 Pro), auto strategy selects the DeepGEMM executor
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            smoke_test(
                name = "moe_fp8pb_tp2_sm120",
                task_info = "data/model/qwen3_moe/q_r_30b_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )
