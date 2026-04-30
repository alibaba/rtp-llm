"""Auto-generated: smoke cases for suite ``smoke_sm100_dense``.

Derived from monolithic SMOKE_TESTS["smoke_sm100_dense"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'dense_tp1_sm100': {   'task_info': 'data/model/qwen3/q_r_l20a_fp4_tp1_py.json',
                           'smoke_args': '--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 '
                                         '--reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 '
                                         '--concurrency_limit 64 --blockwise_use_fp8_kv_cache 1',
                           'gpu_type': 'SM100_ARM',
                           'platform': 'cuda',
                           'markers': ['smoke', 'cuda', 'SM100_ARM'],
                           'timeout': 600},
    'dense_tp2_sm100': {   'task_info': 'data/model/qwen3/q_r_l20a_fp4_tp2_py.json',
                           'smoke_args': '--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 '
                                         '--reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 '
                                         '--concurrency_limit 64 --blockwise_use_fp8_kv_cache 1',
                           'gpu_type': 'SM100_ARM',
                           'platform': 'cuda',
                           'markers': ['smoke', 'cuda', 'SM100_ARM'],
                           'timeout': 600},
    'fp8_attention_sm100': {   'task_info': 'data/model/qwen3/q_r_block_fp8.json',
                               'smoke_args': '--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 '
                                             '--reserver_runtime_mem_mb 178125 --warm_up 0',
                               'gpu_type': 'SM100_ARM',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'SM100_ARM'],
                               'timeout': 600}}
