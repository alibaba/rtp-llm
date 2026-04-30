"""Auto-generated: smoke cases for suite ``smoke_h20_dense``.

Derived from monolithic SMOKE_TESTS["smoke_h20_dense"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'dense_fp8kv_cudagraph': {   'task_info': 'data/model/qwen25/q_r_new_model_py_fp8_kv_cache.json',
                                 'smoke_args': '--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num '
                                               '1000 --fp8_kv_cache 1 --enable_cuda_graph 1  --disable_flash_infer 1',
                                 'gpu_type': 'H20',
                                 'platform': 'cuda',
                                 'markers': ['smoke', 'cuda', 'H20'],
                                 'timeout': 600},
    'dense_fp8_prequant_tp2': {   'task_info': 'data/model/qwen3/q_r_block_fp8.json',
                                  'smoke_args': '--disable_flash_infer 1 --act_type BF16 --reserver_runtime_mem_mb '
                                                '8192 --tp_size 2 --warm_up 0',
                                  'gpu_type': 'H20',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'H20'],
                                  'timeout': 600},
    'dense_fp8pb_dynamic': {   'task_info': 'data/model/qwen3/q_r_h20.json',
                               'smoke_args': '--disable_flash_infer 1 --quantization FP8_PER_BLOCK --act_type BF16 '
                                             '--warm_up 0',
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600},
    'dense_fp8pt_dynamic': {   'task_info': 'data/model/qwen3/q_r_h20_per_tensor_w13.json',
                               'smoke_args': '--disable_flash_infer 1 --quantization FP8_DYNAMIC_PER_TENSOR --act_type '
                                             'BF16',
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600},
    'dense_override_yarn': {   'task_info': 'data/model/qwen3/q_r_override_yarn.json',
                               'smoke_args': '--reserver_runtime_mem_mb 20000 --json_model_override_args \'{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"beta_slow":1.0,"beta_fast":1.0,"mscale":1.0,"extrapolation_factor":1.0}}\' --seq_size_per_block 64 --act_type BF16 --warm_up 0',
                               'envs': ['LOAD_PYTHON_MODEL=1'],
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600}}
