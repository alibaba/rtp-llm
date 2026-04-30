"""Auto-generated: smoke cases for suite ``smoke_sm100_moe``.

Derived from monolithic SMOKE_TESTS["smoke_sm100_moe"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'moe_deepep_normal_tp2_sm100': {   'task_info': 'data/model/qwen3_moe/q_r_30b_py_tp2_sm100.json',
                                       'smoke_args': '--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 '
                                                     '--fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 '
                                                     '--quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 '
                                                     '--use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 '
                                                     '--world_size 2',
                                       'gpu_type': 'SM100_ARM',
                                       'platform': 'cuda',
                                       'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                       'timeout': 600},
    'moe_nvfp4_deepep_ll_cudagraph_dp2_sm100': {   'task_info': 'data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_ll_cg_sm100_arm.json',
                                                   'smoke_args': "--decode_capture_config '1,2' --warm_up 0 "
                                                                 '--enable_cuda_graph 1 --act_type BF16 --dp_size 2 '
                                                                 '--world_size 2 --ep_size 2 --reserver_runtime_mem_mb '
                                                                 '20000 --fp8_kv_cache 1 --seq_size_per_block 64 '
                                                                 '--concurrency_limit 64 --blockwise_use_fp8_kv_cache '
                                                                 '1 --use_deepep_moe 1 --use_deepep_low_latency 1',
                                                   'gpu_type': 'SM100_ARM',
                                                   'platform': 'cuda',
                                                   'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                                   'timeout': 600},
    'moe_nvfp4_deepep_ll_tp2_sm100': {   'task_info': 'data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_ll_sm100_arm.json',
                                         'smoke_args': '--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 '
                                                       '--ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 '
                                                       '--seq_size_per_block 64 --concurrency_limit 64 '
                                                       '--blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 '
                                                       '--use_deepep_low_latency 1',
                                         'gpu_type': 'SM100_ARM',
                                         'platform': 'cuda',
                                         'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                         'timeout': 600},
    'moe_nvfp4_deepep_normal_dp2_sm100': {   'task_info': 'data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_normal_sm100_arm.json',
                                             'smoke_args': '--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 '
                                                           '--ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache '
                                                           '1 --seq_size_per_block 64 --concurrency_limit 64 '
                                                           '--blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 '
                                                           '--use_deepep_low_latency 0 --use_all_gather 0',
                                             'gpu_type': 'SM100_ARM',
                                             'platform': 'cuda',
                                             'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                             'timeout': 600},
    'moe_nvfp4_deepep_normal_tp2_sm100': {   'task_info': 'data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_normal_sm100_arm.json',
                                             'smoke_args': '--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 '
                                                           '--ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache '
                                                           '1 --seq_size_per_block 64 --concurrency_limit 64 '
                                                           '--blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 '
                                                           '--use_deepep_low_latency 0 --use_all_gather 0',
                                             'gpu_type': 'SM100_ARM',
                                             'platform': 'cuda',
                                             'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                             'timeout': 600},
    'next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100': {   'task_info': 'data/model/qwen35/q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json',
                                                        'smoke_args': "--decode_capture_config '1,2,3,4' --warm_up 0 "
                                                                      '--enable_cuda_graph 1 --act_type BF16 --dp_size '
                                                                      '2 --world_size 2 --ep_size 2 '
                                                                      '--reserver_runtime_mem_mb 20000 '
                                                                      '--seq_size_per_block 2048 --concurrency_limit '
                                                                      '64  --kernel_seq_size_per_block 64 '
                                                                      '--use_deepep_moe 1 --use_deepep_low_latency 1',
                                                        'gpu_type': 'SM100_ARM',
                                                        'platform': 'cuda',
                                                        'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                                        'timeout': 600},
    'next_moe_nvfp4_cudagraph_tp2_sm100': {   'task_info': 'data/model/qwen35/q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json',
                                              'smoke_args': "--decode_capture_config '1,2,3,4' --warm_up 0 "
                                                            '--enable_cuda_graph 1 --act_type BF16 --tp_size 2 '
                                                            '--world_size 2 --ep_size 2 --reserver_runtime_mem_mb '
                                                            '20000 --seq_size_per_block 2048 --concurrency_limit 64  '
                                                            '--kernel_seq_size_per_block 64',
                                              'gpu_type': 'SM100_ARM',
                                              'platform': 'cuda',
                                              'markers': ['smoke', 'cuda', 'SM100_ARM'],
                                              'timeout': 600}}
