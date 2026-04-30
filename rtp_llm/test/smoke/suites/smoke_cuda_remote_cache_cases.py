"""Auto-generated: smoke cases for suite ``smoke_cuda_remote_cache``.

Derived from monolithic SMOKE_TESTS["smoke_cuda_remote_cache"]. Edit case definitions here;
do NOT add framework helpers (those live in rtp_llm.test.smoke_framework).
"""

SMOKE_CASES = {   'remote_cache_basic': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache.json',
                              'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 '
                                            '--write_cache_sync 1 --enable_remote_cache true --enable_device_cache 0',
                              'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG'],
                              'gpu_type': 'L20',
                              'platform': 'cuda',
                              'markers': ['smoke', 'cuda', 'L20'],
                              'timeout': 600},
    'remote_cache_basic_async': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache.json',
                                    'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 '
                                                  '--enable_remote_cache true --enable_device_cache 0',
                                    'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG'],
                                    'gpu_type': 'L20',
                                    'platform': 'cuda',
                                    'markers': ['smoke', 'cuda', 'L20'],
                                    'timeout': 600,
                                    'sleep_time_qr': 10},
    'remote_cache_kill': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_kill_remote.json',
                             'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 '
                                           '--enable_remote_cache true --enable_device_cache 0',
                             'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG'],
                             'gpu_type': 'L20',
                             'platform': 'cuda',
                             'markers': ['smoke', 'cuda', 'L20'],
                             'timeout': 600,
                             'sleep_time_qr': 10,
                             'kill_remote': True},
    'remote_cache_tp2': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_tpsize_2.json',
                            'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 '
                                          '--tp_size 2 --enable_remote_cache true --enable_device_cache 0 '
                                          '--reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 '
                                          '--reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000',
                            'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG'],
                            'gpu_type': 'L20',
                            'platform': 'cuda',
                            'markers': ['smoke', 'cuda', 'L20'],
                            'timeout': 600,
                            'sleep_time_qr': 20},
    'remote_cache_pd': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_pd_sep.json',
                           'smoke_args': {   'prefill': '--warm_up 0  --reuse_cache 1 --role_type PREFILL --act_type '
                                                        'FP16 --seq_size_per_block 8 --enable_remote_cache true '
                                                        '--enable_device_cache 0 --reco_put_timeout_ms 12000 '
                                                        '--reco_get_timeout_ms 12000 --reco_get_broadcast_timeout '
                                                        '15000 --reco_put_broadcast_timeout 15000',
                                             'decode': '--warm_up 0  --reuse_cache 1 --role_type DECODE --act_type '
                                                       'FP16 --seq_size_per_block 8 --enable_remote_cache true '
                                                       '--enable_device_cache 0 --reco_put_timeout_ms 12000 '
                                                       '--reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 '
                                                       '--reco_put_broadcast_timeout 15000'},
                           'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG'],
                           'gpu_type': 'L20',
                           'platform': 'cuda',
                           'markers': ['smoke', 'cuda', 'L20'],
                           'timeout': 600,
                           'sleep_time_qr': 20},
    'remote_cache_match_fail': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_match_failure.json',
                                   'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 '
                                                 '--enable_remote_cache true --enable_device_cache 0',
                                   'envs': [   'SEQ_SIZE_PER_BLOCK=8',
                                               'KVCM_LOG_LEVEL=DEBUG',
                                               'ENABLE_DEBUG_SERVICE=TRUE',
                                               'TEST_MATCH_FAILURE=1'],
                                   'gpu_type': 'L20',
                                   'platform': 'cuda',
                                   'markers': ['smoke', 'cuda', 'L20'],
                                   'timeout': 600,
                                   'sleep_time_qr': 10},
    'remote_cache_write_start_fail': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json',
                                         'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 '
                                                       '--seq_size_per_block 8 --enable_remote_cache true '
                                                       '--enable_device_cache 0',
                                         'envs': [   'SEQ_SIZE_PER_BLOCK=8',
                                                     'KVCM_LOG_LEVEL=DEBUG',
                                                     'ENABLE_DEBUG_SERVICE=TRUE',
                                                     'TEST_START_WRITE_FAILURE=1'],
                                         'gpu_type': 'L20',
                                         'platform': 'cuda',
                                         'markers': ['smoke', 'cuda', 'L20'],
                                         'timeout': 600,
                                         'sleep_time_qr': 10},
    'remote_cache_write_finish_fail': {   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json',
                                          'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 '
                                                        '--seq_size_per_block 8 --enable_remote_cache true '
                                                        '--enable_device_cache 0',
                                          'envs': [   'SEQ_SIZE_PER_BLOCK=8',
                                                      'KVCM_LOG_LEVEL=DEBUG',
                                                      'ENABLE_DEBUG_SERVICE=TRUE',
                                                      'TEST_FINISH_WRITE_FAILURE=1'],
                                          'gpu_type': 'L20',
                                          'platform': 'cuda',
                                          'markers': ['smoke', 'cuda', 'L20'],
                                          'timeout': 600,
                                          'sleep_time_qr': 10},
    'remote_cache_edge': {   'task_info': 'data/model/qwen25/q_r_l20_cache_edge_case_1_remote_cache.json',
                             'smoke_args': '--warm_up 0  --reuse_cache 1 --act_type FP16 --seq_size_per_block 4 '
                                           '--enable_remote_cache true --enable_device_cache 0',
                             'envs': ['SEQ_SIZE_PER_BLOCK=4', 'KVCM_LOG_LEVEL=DEBUG'],
                             'gpu_type': 'L20',
                             'platform': 'cuda',
                             'markers': ['smoke', 'cuda', 'L20'],
                             'timeout': 600,
                             'sleep_time_qr': 10}}
