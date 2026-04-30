"""Auto-generated: smoke cases for suite ``smoke_rocm_eagle``.

Derived from monolithic SMOKE_TESTS["smoke_rocm_eagle"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'rocm_eagle_qwen2_14b': {   'task_info': 'data/model/qwen2_14b/q_r_mtp_rocm.json',
                                'smoke_args': '--max_seq_len 16384 --tp_size 1 --use_asm_pa 1 --ft_disable_custom_ar 1 '
                                              '--sp_type eagle --gen_num_per_cycle 4 --warm_up 0 --act_type BF16 '
                                              '--sp_model_type qwen_2-mtp --sp_checkpoint_path '
                                              '/mnt/nas1/mtp_reg/qwen2_14b_draft/ --reserver_runtime_mem_mb 4002',
                                'gpu_type': 'MI308X-ROCM7',
                                'platform': 'cuda',
                                'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                'timeout': 600}}
