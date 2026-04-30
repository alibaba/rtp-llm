"""Auto-generated: smoke cases for suite ``smoke_rocm_embedding``.

Derived from monolithic SMOKE_TESTS["smoke_rocm_embedding"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'rocm_embedding_qwen3_32b_ptpc_fp8': {   'task_info': 'data/model/qwen3/ptpc_q_r_fp8_py.json',
                                             'smoke_args': '--reserver_runtime_mem_mb 107813 --use_aiter_pa 1 '
                                                           '--seq_size_per_block 16 --fp8_kv_cache 1',
                                             'gpu_type': 'MI308X-ROCM7',
                                             'platform': 'cuda',
                                             'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                             'timeout': 600},
    'rocm_embedding_bert_st': {   'task_info': 'data/model/bert/q_r.json',
                                  'smoke_args': '--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type '
                                                'FP16',
                                  'gpu_type': 'MI308X-ROCM7',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                  'timeout': 600},
    'rocm_embedding_roberta_st': {   'task_info': 'data/model/bert/roberta_q_r.json',
                                     'smoke_args': '--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type '
                                                   'FP16',
                                     'gpu_type': 'MI308X-ROCM7',
                                     'platform': 'cuda',
                                     'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                     'timeout': 600},
    'rocm_embedding_roberta_sparse': {   'task_info': 'data/model/bert/sparse_roberta_q_r.json',
                                         'smoke_args': '--task_type SPARSE_EMBEDDING --seq_size_per_block 16 '
                                                       '--use_aiter_pa 1 --use_asm_pa 1 --act_type FP16',
                                         'gpu_type': 'MI308X-ROCM7',
                                         'platform': 'cuda',
                                         'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                         'timeout': 600},
    'rocm_embedding_roberta_colbert': {   'task_info': 'data/model/bert/colbert_roberta_q_r.json',
                                          'smoke_args': '--task_type COLBERT_EMBEDDING --seq_size_per_block 16 '
                                                        '--use_aiter_pa 1 --use_asm_pa 1 --act_type FP16',
                                          'gpu_type': 'MI308X-ROCM7',
                                          'platform': 'cuda',
                                          'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                          'timeout': 600},
    'rocm_embedding_bert_classifier': {   'task_info': 'data/model/bert/bert_classifier_q_r.json',
                                          'smoke_args': '--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 '
                                                        '--act_type FP16',
                                          'gpu_type': 'MI308X-ROCM7',
                                          'platform': 'cuda',
                                          'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                          'timeout': 600},
    'rocm_embedding_roberta_reranker': {   'task_info': 'data/model/bert/reranker_q_r.json',
                                           'smoke_args': '--task_type RERANKER --max_context_batch_size 10 '
                                                         '--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 '
                                                         '--act_type FP16',
                                           'gpu_type': 'MI308X-ROCM7',
                                           'platform': 'cuda',
                                           'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                           'timeout': 600},
    'rocm_embedding_roberta_truncate': {   'task_info': 'data/model/bert/reranker_q_r_base.json',
                                           'smoke_args': '--task_type RERANKER --max_context_batch_size 10 --act_type '
                                                         'FP16 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1',
                                           'gpu_type': 'MI308X-ROCM7',
                                           'platform': 'cuda',
                                           'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                           'timeout': 600},
    'rocm_embedding_bge_reranker_trt_fmha': {   'task_info': 'data/model/bert/classifier_q_r.json',
                                                'smoke_args': '--enable_trt_fmha 0 --enable_open_source_fmha 0 '
                                                              '--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 '
                                                              '--act_type FP16',
                                                'gpu_type': 'MI308X-ROCM7',
                                                'platform': 'cuda',
                                                'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                                'timeout': 600},
    'rocm_embedding_qwen3_32b_ptpc_fp8_cudagraph': {   'task_info': 'data/model/qwen3/ptpc_q_r_fp8_py.json',
                                                       'smoke_args': '--enable_cuda_graph 1 --reserver_runtime_mem_mb '
                                                                     '107813 --use_aiter_pa 1 --seq_size_per_block 16 '
                                                                     '--fp8_kv_cache 1',
                                                       'gpu_type': 'MI308X-ROCM7',
                                                       'platform': 'cuda',
                                                       'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                                       'timeout': 600}}
