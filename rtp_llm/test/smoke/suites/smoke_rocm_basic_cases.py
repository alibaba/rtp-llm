"""Auto-generated: smoke cases for suite ``smoke_rocm_basic``.

Derived from monolithic SMOKE_TESTS["smoke_rocm_basic"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'rocm_basic_cache_reuse': {   'task_info': 'data/model/qwen2/q_r_reuse.json',
                                  'smoke_args': '--reuse_cache 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa '
                                                '1 --act_type FP16',
                                  'gpu_type': 'MI308X-ROCM7',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                  'timeout': 600},
    'rocm_basic_batch_cache_reuse': {   'task_info': 'data/model/qwen3/q_r_308x_batch_cache.json',
                                        'smoke_args': '--reuse_cache 1 --enable_cuda_graph 1 --seq_size_per_block 16 '
                                                      '--use_aiter_pa 1 --use_asm_pa 1 --act_type FP16',
                                        'gpu_type': 'MI308X-ROCM7',
                                        'platform': 'cuda',
                                        'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                        'timeout': 600}}
