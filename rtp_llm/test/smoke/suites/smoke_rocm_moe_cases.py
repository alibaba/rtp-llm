"""Auto-generated: smoke cases for suite ``smoke_rocm_moe``.

Derived from monolithic SMOKE_TESTS["smoke_rocm_moe"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'rocm_moe_qwen3_30b_basic': {   'task_info': 'data/model/qwen3_moe/q_r_30b_amd_py.json',
                                    'smoke_args': '--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type '
                                                  'BF16 --reserver_runtime_mem_mb 51200 --tp_size 1 --world_size 1 '
                                                  '--ep_size 1',
                                    'gpu_type': 'MI308X-ROCM7',
                                    'platform': 'cuda',
                                    'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                    'timeout': 600},
    'rocm_moe_qwen3_30b_tp2': {   'task_info': 'data/model/qwen3_moe/q_r_30b_amd_py_tp2.json',
                                  'smoke_args': '--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type '
                                                'BF16 --reserver_runtime_mem_mb 51200 --tp_size 2 --world_size 2 '
                                                '--ep_size 1',
                                  'gpu_type': 'MI308X-ROCM7',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                  'timeout': 600}}
