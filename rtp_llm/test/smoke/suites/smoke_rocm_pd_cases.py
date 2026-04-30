"""Auto-generated: smoke cases for suite ``smoke_rocm_pd``.

Derived from monolithic SMOKE_TESTS["smoke_rocm_pd"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'rocm_pd_qwen3_8b': {   'task_info': 'data/model/qwen3/q_r_new_model_py.json',
                            'smoke_args': {   'prefill': '--test_block_num 10 --warm_up 0 --seq_size_per_block 16 '
                                                         '--act_type bf16 --use_swizzleA 1 --use_asm_pa 1 '
                                                         '--disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 '
                                                         '--role_type PREFILL --world_size 1',
                                              'decode': '--test_block_num 10 --warm_up 0 --seq_size_per_block 16 '
                                                        '--act_type bf16 --use_swizzleA 1 --use_asm_pa 1 '
                                                        '--disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 '
                                                        '--role_type DECODE --world_size 1'},
                            'gpu_type': 'MI308X-ROCM7',
                            'platform': 'cuda',
                            'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                            'timeout': 600}}
