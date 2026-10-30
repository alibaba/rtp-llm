# RTP-LLM 对齐与埋点覆盖清单

唯一线上来源：[Whale-LLM-New【禁止修改】v514](https://grafana.alibaba-inc.com/d/iZ5Q_81nz)。本清单记录默认 View 使用的原面板；不再使用独立 FlexLB 旧大盘。

状态表示源码中是否有对应导出，不表示每次实验必有样本。待埋点项仅保留查询/说明，不增加虚假指标；资源项因共享 JVM 归属未接线，暂不绑定每 Engine 查询。

| 角色 | RTP-LLM panel | 标题 | 新框架导出 | 原指标 |
|---|---:|---|---|---|
| prefill | 10379 | prefill tps | 已上报 | rtp_llm_context_tps |
| prefill | 10368 | prefill wall tps | 已上报 | rtp_llm_context_wall_tps |
| prefill | 10381 | input wall tps | 已上报 | rtp_llm_context_wall_tps |
| prefill | 218 | timeout qps | 待埋点 | rtp_llm_framework_qps |
| prefill | 10373 | total tps | 已上报 | rtp_llm_total_tps |
| prefill | 10370 | prefill tps with cache | 已上报 | rtp_llm_context_tps_with_cache |
| prefill | 10372 | prefill wall tps with cache | 已上报 | rtp_llm_context_wall_tps_with_cache |
| prefill | 10382 | input wall tps with catch | 已上报 | rtp_llm_context_wall_tps_with_cache |
| prefill | 231 | model forward rt | 已上报 | rtp_llm_model_forward_us |
| prefill | 10081 | wall tps report interval | 已上报 | rtp_llm_wall_tps_report_interval_us |
| prefill | 159 | gather model input rt | 待埋点 | rtp_llm_gather_model_input_us |
| prefill | 215 | 请求队列排队延迟 | 待埋点 | rtp_llm_wait_latency_us |
| prefill | 234 | first token latency | 待埋点 | rtp_llm_first_token_latency_us |
| prefill | 121 | framework qps | 已上报 | rtp_llm_framework_qps |
| prefill | 162 | framework cancel qps | 待埋点 | rtp_llm_rpc_cancel_qps |
| prefill | 130 | total latency | 待埋点 | rtp_llm_latency_us |
| prefill | 131 | pause latency | 待埋点 | rtp_llm_pause_latency_us |
| prefill | 233 | dispatch output rt | 待埋点 | rtp_llm_dispatch_output_us_metric |
| prefill | 230 | tp sync input rt | 待埋点 | rtp_llm_tp_sync_input_us |
| prefill | 160 | step latency | 待埋点 | rtp_llm_step_latency_us |
| prefill | 161 | framework error qps | 已上报 | rtp_llm_framework_error_qps |
| prefill | 217 | 请求超时时间 | 待埋点 | rtp_llm_timeout_lantency_us |
| prefill | 10587 | stream canrun to running latency | 待埋点 | rtp_llm_stream_canrun_to_running_us |
| prefill | 10588 | stream loading cache latency | 待埋点 | rtp_llm_stream_loading_cache_latency_us |
| prefill | 10098 | model inputs dump latency | 待埋点 | rtp_llm_model_inputs_log_us |
| prefill | 232 | sample rt | 待埋点 | rtp_llm_sample_input_us |
| prefill | 10586 | py framework error qps | 待埋点 | py_rtp_master_queue_reject_qps |
| prefill | 10589 | stream enqueue to canrun latency | 待埋点 | rtp_llm_stream_enqueue_to_canrun_us |
| prefill | 10629 | stream update rt | 待埋点 | rtp_llm_stream_update_latency_us |
| prefill | 10630 | xgrammar timeout qps | 待埋点 | rtp_llm_grammar_update_timeout_qps |
| prefill | 128 | running stream size | 已上报 | rtp_llm_running_stream_size |
| prefill | 220 | context_batch_size_when_has_context | 待埋点 | rtp_llm_context_batch_size_when_has_context |
| prefill | 150 | query batch size | 待埋点 | rtp_llm_query_batch_size |
| prefill | 327 | remote running stream size | 待埋点 | rtp_llm_remote_running_stream_size |
| prefill | 122 | context batch size | 待埋点 | rtp_llm_context_batch_size |
| prefill | 127 | waiting stream size | 已上报 | rtp_llm_wait_stream_size |
| prefill | 124 | batch max seq len | 待埋点 | rtp_llm_max_seq_len |
| prefill | 226 | batch max seq len when has context | 待埋点 | rtp_llm_max_seq_len_when_has_context |
| prefill | 125 | input token length | 待埋点 | rtp_llm_input_token_length |
| prefill | 10099 | context batch size / max agg | 待埋点 | rtp_llm_context_batch_size |
| prefill | 10163 | waiting stream size max/agg | 已上报 | rtp_llm_wait_stream_size |
| prefill | 228 | batch with prefill times | 待埋点 | rtp_llm_batch_with_prefill_times |
| prefill | 229 | batch with prefill lens | 待埋点 | rtp_llm_batch_with_prefill_len |
| prefill | 227 | execute token size when has context | 待埋点 | rtp_llm_execute_token_size_when_has_context |
| prefill | 146 | execute token size | 待埋点 | rtp_llm_execute_token_size |
| prefill | 170 | get rpc connection rt | 待埋点 | rtp_llm_rpc_get_rpc_connection_rt_us |
| prefill | 169 | multimodal process rt | 待埋点 | rtp_llm_rpc_multimodal_process_rt_us |
| prefill | 223 | remote allocate resource rt | 待埋点 | rtp_llm_rpc_remote_allocate_resource_rt_us |
| prefill | 171 | enqueue request rt | 待埋点 | rtp_llm_rpc_enqueue_request_rt_us |
| prefill | 186 | remote load cache start rt | 待埋点 | rtp_llm_rpc_remote_load_cache_start_rt_us |
| prefill | 224 | remote load cache end rt | 待埋点 | rtp_llm_rpc_remote_load_cache_end_rt_us |
| prefill | 187 | poll local output rt | 待埋点 | rtp_llm_rpc_poll_local_output_rt_us |
| prefill | 189 | remote generate rt | 待埋点 | rtp_llm_rpc_remote_generate_rt_us |
| prefill | 10108 | load cache gap time | 待埋点 | rtp_llm.cache_store.load.server.transfer_gap_latency_us |
| prefill | 10105 | write latency | 待埋点 | rtp_llm.cache_store.load.server.write.latency_us |
| prefill | 10106 | first block ready cost | 待埋点 | rtp_llm.cache_store.first_block_cost_us_ |
| prefill | 10107 | all blocks ready cost | 待埋点 | rtp_llm.cache_store.load.server.all_block_ready_latency_us |
| prefill | 197 | poll remote output rt | 待埋点 | rtp_llm_rpc_poll_remote_output_rt_us |
| prefill | 10035 | Prefill 理论命中/总输入数量 | 待埋点 | rtp_llm_prefill_worker_recent_cache_key_hit_count,rtp_llm_prefill_worker_recent_cache_key_total_count |
| prefill | 143 | kv cache item num | 待埋点 | rtp_llm_kv_cache_item_num |
| prefill | 147 | 实际的总reuse len | 待埋点 | rtp_llm_kv_cache_reuse_length |
| prefill | 10636 | new device reuse length | 待埋点 | rtp_llm_stream_cache_device_reuse_length |
| prefill | 491 | new memory reuse length | 待埋点 | rtp_llm_stream_cache_memory_reuse_length |
| prefill | 10590 | new disk reuse length | 待埋点 | rtp_llm_stream_cache_disk_reuse_length |
| prefill | 510 | new remote reuse length | 待埋点 | rtp_llm_stream_cache_remote_reuse_length |
| prefill | 10094 | kv cache request ref blocks | 待埋点 | rtp_llm_kv_cache_request_ref_blocks |
| prefill | 174 | kv cache free blocks | 已上报 | rtp_llm_kv_cache_free_blocks |
| prefill | 173 | kv cache available blocks | 已上报 | rtp_llm_kv_cache_available_blocks |
| prefill | 154 | kv cache used ratio | 已上报 | rtp_llm_kv_cache_used_ratio |
| prefill | 558 | kv_cache_pool_used_ratio分池最大值 | 部分待埋点 | rtp_llm_kv_cache_pool_used_ratio,rtp_llm_kv_cache_used_ratio |
| prefill | 142 | kv cache available token num | 已上报 | rtp_llm_kv_cache_left_seq |
| prefill | 10095 | kv cache connector ref blocks | 待埋点 | rtp_llm_kv_cache_connector_ref_blocks |
| prefill | 222 | match_cost_time | 待埋点 | rtp_llm_match_cost_time_us |
| prefill | 235 | mr cost time | 待埋点 | rtp_llm_mr_cost_time_ms |
| prefill | 10600 | eviction ref blocks | 待埋点 | rtp_llm_kv_cache_eviction_target_ref_blocks |
| prefill | 10086 | evicted block lifetime | 待埋点 | rtp_llm_kv_cache_evicted_block_lifetime_ms |
| prefill | 552 | kv cache available blocks | 部分待埋点 | rtp_llm_kv_cache_pool_available_blocks,rtp_llm_kv_cache_available_blocks |
| prefill | 553 | kv cache free blocks | 部分待埋点 | rtp_llm_kv_cache_pool_free_blocks,rtp_llm_kv_cache_free_blocks |
| prefill | 554 | request_ref_blocks | 待埋点 | rtp_llm_kv_cache_pool_request_ref_blocks,rtp_llm_kv_cache_request_ref_blocks |
| prefill | 10631 | remote_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_remote_connector_ref_blocks |
| prefill | 10632 | pd_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_pd_connector_ref_blocks |
| prefill | 556 | kv cache total blocks | 部分待埋点 | rtp_llm_kv_cache_pool_total_blocks,rtp_llm_kv_cache_total_blocks |
| prefill | 557 | kv_cache_pool_used_ratio | 部分待埋点 | rtp_llm_kv_cache_pool_used_ratio,rtp_llm_kv_cache_used_ratio |
| prefill | 555 | kv_cache_pool_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_pool_connector_ref_blocks |
| prefill | 10593 | kv_cache_pool_load_ref_blocks | 待埋点 | rtp_llm_kv_cache_load_ref_blocks |
| prefill | 10594 | kv_cache_pool_store_ref_blocks | 待埋点 | rtp_llm_kv_cache_store_ref_blocks |
| prefill | 10602 | kv_cache_pool_eviction_ref_blocks | 待埋点 | rtp_llm_kv_cache_eviction_target_ref_blocks |
| prefill | 10635 | rtp_llm_kv_cache_frozen_ref_blocks | 待埋点 | rtp_llm_kv_cache_frozen_ref_blocks |
| prefill | 420 | match qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_qps |
| prefill | 423 | match latency | 待埋点 | rtp_llm_kv_cache_memory_cache_match_latency_us |
| prefill | 422 | match miss qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_failed_qps |
| prefill | 421 | match ignore qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_none_qps |
| prefill | 424 | match input token | 待埋点 | rtp_llm_kv_cache_memory_cache_match_input_token |
| prefill | 425 | real matched token | 待埋点 | rtp_llm_kv_cache_memory_cache_matched_token |
| prefill | 426 | read qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_qps |
| prefill | 487 | read failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_failed_qps |
| prefill | 489 | read none qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_none_qps |
| prefill | 488 | read latency | 待埋点 | rtp_llm_kv_cache_memory_cache_read_latency_us |
| prefill | 486 | read input token | 待埋点 | rtp_llm_kv_cache_memory_cache_read_input_token |
| prefill | 490 | real read token | 待埋点 | rtp_llm_kv_cache_memory_cache_read_token |
| prefill | 485 | write qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_qps |
| prefill | 427 | write failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_failed_qps |
| prefill | 437 | write none qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_none_qps |
| prefill | 428 | write cache latency | 待埋点 | rtp_llm_kv_cache_memory_cache_write_latency_us |
| prefill | 429 | write input token | 待埋点 | rtp_llm_kv_cache_memory_cache_write_input_token |
| prefill | 430 | real write token | 待埋点 | rtp_llm_kv_cache_memory_cache_write_token |
| prefill | 431 | copy cache qps | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_qps |
| prefill | 432 | copy failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_failed_qps |
| prefill | 433 | memory copy latency | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_latency_us |
| prefill | 434 | memory total block num | 部分待埋点 | rtp_llm_kv_cache_memory_cache_status_total_block_num,rtp_llm_kv_cache_total_blocks |
| prefill | 435 | memory allocated block num | 部分待埋点 | rtp_llm_kv_cache_memory_cache_status_allocated_block_num,rtp_llm_kv_cache_used_blocks |
| prefill | 436 | memory available block num | 待埋点 | rtp_llm_kv_cache_memory_cache_status_available_block_num,rtp_llm_kv_cache_available_blocks |
| prefill | 10087 | memory used ratio | 待埋点 | rtp_llm_kv_cache_memory_cache_status_used_ratio |
| prefill | 10418 | Memory copy queue wait | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_task_queue_wait_us |
| prefill | 10422 | Memory copy pool submit failed QPS | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_submit_failed_qps |
| prefill | 10420 | Memory copy pool active threads | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_active_threads |
| prefill | 10421 | Memory copy pool pending tasks | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_pending_tasks |
| prefill | 10424 | Memory copy throughput | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_throughput_bytes_per_second |
| decode | 10369 | decode tps | 已上报 | rtp_llm_generate_tps |
| decode | 10383 | output tps | 已上报 | rtp_llm_generate_tps |
| decode | 218 | timeout qps | 待埋点 | rtp_llm_framework_qps |
| decode | 10373 | total tps | 已上报 | rtp_llm_total_tps |
| decode | 231 | model forward rt | 已上报 | rtp_llm_model_forward_us |
| decode | 10081 | wall tps report interval | 已上报 | rtp_llm_wall_tps_report_interval_us |
| decode | 159 | gather model input rt | 待埋点 | rtp_llm_gather_model_input_us |
| decode | 215 | 请求队列排队延迟 | 待埋点 | rtp_llm_wait_latency_us |
| decode | 234 | first token latency | 待埋点 | rtp_llm_first_token_latency_us |
| decode | 121 | framework qps | 已上报 | rtp_llm_framework_qps |
| decode | 162 | framework cancel qps | 待埋点 | rtp_llm_rpc_cancel_qps |
| decode | 130 | total latency | 待埋点 | rtp_llm_latency_us |
| decode | 131 | pause latency | 待埋点 | rtp_llm_pause_latency_us |
| decode | 233 | dispatch output rt | 待埋点 | rtp_llm_dispatch_output_us_metric |
| decode | 230 | tp sync input rt | 待埋点 | rtp_llm_tp_sync_input_us |
| decode | 160 | step latency | 待埋点 | rtp_llm_step_latency_us |
| decode | 161 | framework error qps | 已上报 | rtp_llm_framework_error_qps |
| decode | 217 | 请求超时时间 | 待埋点 | rtp_llm_timeout_lantency_us |
| decode | 10587 | stream canrun to running latency | 待埋点 | rtp_llm_stream_canrun_to_running_us |
| decode | 10588 | stream loading cache latency | 待埋点 | rtp_llm_stream_loading_cache_latency_us |
| decode | 10098 | model inputs dump latency | 待埋点 | rtp_llm_model_inputs_log_us |
| decode | 232 | sample rt | 待埋点 | rtp_llm_sample_input_us |
| decode | 10112 | TPOT(估计值) | 待埋点 | rtp_llm_sp_estimate_tpot_us |
| decode | 10586 | py framework error qps | 待埋点 | py_rtp_master_queue_reject_qps |
| decode | 10589 | stream enqueue to canrun latency | 待埋点 | rtp_llm_stream_enqueue_to_canrun_us |
| decode | 10629 | stream update rt | 待埋点 | rtp_llm_stream_update_latency_us |
| decode | 10630 | xgrammar timeout qps | 待埋点 | rtp_llm_grammar_update_timeout_qps |
| decode | 128 | running stream size | 已上报 | rtp_llm_running_stream_size |
| decode | 221 | generate_batch_size_when_has_context | 待埋点 | rtp_llm_generate_batch_size_when_has_context |
| decode | 149 | iterate count | 待埋点 | rtp_llm_iterate_count |
| decode | 150 | query batch size | 待埋点 | rtp_llm_query_batch_size |
| decode | 327 | remote running stream size | 待埋点 | rtp_llm_remote_running_stream_size |
| decode | 127 | waiting stream size | 已上报 | rtp_llm_wait_stream_size |
| decode | 124 | batch max seq len | 待埋点 | rtp_llm_max_seq_len |
| decode | 226 | batch max seq len when has context | 待埋点 | rtp_llm_max_seq_len_when_has_context |
| decode | 125 | input token length | 待埋点 | rtp_llm_input_token_length |
| decode | 123 | generate batch size | 待埋点 | rtp_llm_generate_batch_size |
| decode | 10163 | waiting stream size max/agg | 已上报 | rtp_llm_wait_stream_size |
| decode | 145 | output token length (AVG 平均值) | 待埋点 | rtp_llm_output_token_length |
| decode | 439 | output token length (Max 最大值) | 待埋点 | rtp_llm_output_token_length |
| decode | 227 | execute token size when has context | 待埋点 | rtp_llm_execute_token_size_when_has_context |
| decode | 146 | execute token size | 待埋点 | rtp_llm_execute_token_size |
| decode | 219 | output token length(timeout) | 待埋点 | rtp_llm_output_token_length |
| decode | 192 | prepare generate context rt | 待埋点 | rtp_llm_rpc_prepare_generate_context_rt_us |
| decode | 196 | allocate resource rt | 待埋点 | rtp_llm_rpc_allocate_resource_rt_us |
| decode | 194 | load cache from prefill rt | 待埋点 | rtp_llm_rpc_load_cache_from_prefill_rt_us |
| decode | 211 | load cache min rt | 待埋点 | rtp_llm_rpc_load_cache_min_rt_us |
| decode | 212 | load cache max rt | 待埋点 | rtp_llm_rpc_load_cache_max_rt_us |
| decode | 213 | load cache polling cost | 待埋点 | rtp_llm_rpc_load_cache_polling_cost_us |
| decode | 193 | local generate rt | 待埋点 | rtp_llm_rpc_local_generate_rt_us |
| decode | 10114 | load block count | 待埋点 | rtp_llm.cache_store.load.client.block_count |
| decode | 10115 | load block size | 待埋点 | rtp_llm.cache_store.load.client.total_block_size |
| decode | 143 | kv cache item num | 待埋点 | rtp_llm_kv_cache_item_num |
| decode | 10094 | kv cache request ref blocks | 待埋点 | rtp_llm_kv_cache_request_ref_blocks |
| decode | 174 | kv cache free blocks | 已上报 | rtp_llm_kv_cache_free_blocks |
| decode | 173 | kv cache available blocks | 已上报 | rtp_llm_kv_cache_available_blocks |
| decode | 154 | kv cache used ratio | 已上报 | rtp_llm_kv_cache_used_ratio |
| decode | 558 | kv_cache_pool_used_ratio分池最大值 | 部分待埋点 | rtp_llm_kv_cache_pool_used_ratio,rtp_llm_kv_cache_used_ratio |
| decode | 142 | kv cache available token num | 已上报 | rtp_llm_kv_cache_left_seq |
| decode | 10095 | kv cache connector ref blocks | 待埋点 | rtp_llm_kv_cache_connector_ref_blocks |
| decode | 222 | match_cost_time | 待埋点 | rtp_llm_match_cost_time_us |
| decode | 235 | mr cost time | 待埋点 | rtp_llm_mr_cost_time_ms |
| decode | 10600 | eviction ref blocks | 待埋点 | rtp_llm_kv_cache_eviction_target_ref_blocks |
| decode | 10086 | evicted block lifetime | 待埋点 | rtp_llm_kv_cache_evicted_block_lifetime_ms |
| decode | 552 | kv cache available blocks | 部分待埋点 | rtp_llm_kv_cache_pool_available_blocks,rtp_llm_kv_cache_available_blocks |
| decode | 553 | kv cache free blocks | 部分待埋点 | rtp_llm_kv_cache_pool_free_blocks,rtp_llm_kv_cache_free_blocks |
| decode | 554 | request_ref_blocks | 待埋点 | rtp_llm_kv_cache_pool_request_ref_blocks,rtp_llm_kv_cache_request_ref_blocks |
| decode | 10631 | remote_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_remote_connector_ref_blocks |
| decode | 10632 | pd_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_pd_connector_ref_blocks |
| decode | 556 | kv cache total blocks | 部分待埋点 | rtp_llm_kv_cache_pool_total_blocks,rtp_llm_kv_cache_total_blocks |
| decode | 557 | kv_cache_pool_used_ratio | 部分待埋点 | rtp_llm_kv_cache_pool_used_ratio,rtp_llm_kv_cache_used_ratio |
| decode | 555 | kv_cache_pool_connector_ref_blocks | 待埋点 | rtp_llm_kv_cache_pool_connector_ref_blocks |
| decode | 10593 | kv_cache_pool_load_ref_blocks | 待埋点 | rtp_llm_kv_cache_load_ref_blocks |
| decode | 10594 | kv_cache_pool_store_ref_blocks | 待埋点 | rtp_llm_kv_cache_store_ref_blocks |
| decode | 10602 | kv_cache_pool_eviction_ref_blocks | 待埋点 | rtp_llm_kv_cache_eviction_target_ref_blocks |
| decode | 10635 | rtp_llm_kv_cache_frozen_ref_blocks | 待埋点 | rtp_llm_kv_cache_frozen_ref_blocks |
| decode | 420 | match qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_qps |
| decode | 423 | match latency | 待埋点 | rtp_llm_kv_cache_memory_cache_match_latency_us |
| decode | 422 | match miss qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_failed_qps |
| decode | 421 | match ignore qps | 待埋点 | rtp_llm_kv_cache_memory_cache_match_none_qps |
| decode | 424 | match input token | 待埋点 | rtp_llm_kv_cache_memory_cache_match_input_token |
| decode | 425 | real matched token | 待埋点 | rtp_llm_kv_cache_memory_cache_matched_token |
| decode | 426 | read qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_qps |
| decode | 487 | read failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_failed_qps |
| decode | 489 | read none qps | 待埋点 | rtp_llm_kv_cache_memory_cache_read_none_qps |
| decode | 488 | read latency | 待埋点 | rtp_llm_kv_cache_memory_cache_read_latency_us |
| decode | 486 | read input token | 待埋点 | rtp_llm_kv_cache_memory_cache_read_input_token |
| decode | 490 | real read token | 待埋点 | rtp_llm_kv_cache_memory_cache_read_token |
| decode | 485 | write qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_qps |
| decode | 427 | write failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_failed_qps |
| decode | 437 | write none qps | 待埋点 | rtp_llm_kv_cache_memory_cache_write_none_qps |
| decode | 428 | write cache latency | 待埋点 | rtp_llm_kv_cache_memory_cache_write_latency_us |
| decode | 429 | write input token | 待埋点 | rtp_llm_kv_cache_memory_cache_write_input_token |
| decode | 430 | real write token | 待埋点 | rtp_llm_kv_cache_memory_cache_write_token |
| decode | 431 | copy cache qps | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_qps |
| decode | 432 | copy failed qps | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_failed_qps |
| decode | 433 | memory copy latency | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_latency_us |
| decode | 434 | memory total block num | 部分待埋点 | rtp_llm_kv_cache_memory_cache_status_total_block_num,rtp_llm_kv_cache_total_blocks |
| decode | 435 | memory allocated block num | 部分待埋点 | rtp_llm_kv_cache_memory_cache_status_allocated_block_num,rtp_llm_kv_cache_used_blocks |
| decode | 436 | memory available block num | 待埋点 | rtp_llm_kv_cache_memory_cache_status_available_block_num,rtp_llm_kv_cache_available_blocks |
| decode | 10087 | memory used ratio | 待埋点 | rtp_llm_kv_cache_memory_cache_status_used_ratio |
| decode | 10418 | Memory copy queue wait | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_task_queue_wait_us |
| decode | 10422 | Memory copy pool submit failed QPS | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_submit_failed_qps |
| decode | 10420 | Memory copy pool active threads | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_active_threads |
| decode | 10421 | Memory copy pool pending tasks | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_pool_pending_tasks |
| decode | 10424 | Memory copy throughput | 待埋点 | rtp_llm_kv_cache_memory_cache_copy_throughput_bytes_per_second |
| decode | 546 | TPOT(估计值) | 待埋点 | rtp_llm_sp_estimate_tpot_us |
| decode | 548 | 接受率 | 待埋点 | rtp_llm_sp_avg_accept_rate |
| decode | 549 | 接受率(不算入bonus) | 待埋点 | rtp_llm_sp_avg_fix_accept_rate |
| decode | 402 | 每步平均token数目 | 待埋点 | rtp_llm_sp_avg_accept_token_num |
| decode | 10153 | sp_accept_rate | 待埋点 | py_rtp_frontend_speculative_accept_rate |
| decode | 10155 | sp_total_accepted_token_num | 待埋点 | py_rtp_frontend_speculative_avg_accept_length |
| decode | 178 | sp_step_latency | 待埋点 | rtp_llm_sp_step_latency_us |
| decode | 176 | sp_total_accepted_token_num | 待埋点 | rtp_llm_sp_total_accepted_token_num |
| decode | 182 | sp_total_propose_token_num | 待埋点 | rtp_llm_sp_total_propose_token_num |
| vit | 449 | vit qps | 待埋点 | py_rtp_vit_qps |
| vit | 458 | vit success qps | 待埋点 | py_rtp_vit_success_qps |
| vit | 457 | vit error qps | 待埋点 | py_rtp_vit_error_qps |
| vit | 452 | vit preprocess rt | 待埋点 | py_rtp_vit_download_rt,py_rtp_vit_preprocess_other_rt |
| vit | 10350 | vit download rt | 待埋点 | py_rtp_vit_download_rt |
| vit | 10351 | vit preprocess rt(except download) | 待埋点 | py_rtp_vit_preprocess_other_rt |
| vit | 10365 | vit preprocess waiting queue size | 待埋点 | py_rtp_vit_preprocess_queue_size |
| vit | 10366 | vit images total num | 待埋点 | py_rtp_vit_image_count |
| vit | 456 | vit embedding rt | 待埋点 | py_rtp_vit_embedding_rt |
| vit | 10352 | vit embedding batch rt | 待埋点 | py_rtp_vit_embedding_batch_rt |
| vit | 10353 | vit embedding length | 待埋点 | py_rtp_vit_embedding_length |
| vit | 10354 | vit embedding waiting queue size | 待埋点 | py_rtp_vit_embedding_queue_size |
| vit | 10355 | vit embedding waiting time | 待埋点 | py_rtp_vit_embedding_queue_wait_rt |
| vit | 10356 | vit cuda graph hit qps | 待埋点 | py_rtp_vit_cuda_graph_hit_qps |
| vit | 10357 | vit cuda graph miss qps | 待埋点 | py_rtp_vit_cuda_graph_miss_qps |
| vit | 10358 | vit cuda graph capture qps | 待埋点 | py_rtp_vit_cuda_graph_capture_qps |
| vit | 10359 | vit cuda graph fallback qps | 待埋点 | py_rtp_vit_cuda_graph_fallback_qps |
| vit | 10360 | vit embedding cache hit qps | 待埋点 | py_rtp_vit_embedding_cache_hit_qps |
| vit | 10361 | vit embedding cache miss qps | 待埋点 | py_rtp_vit_embedding_cache_miss_qps |
| vit | 10362 | vit embedding cache inflight qps | 待埋点 | py_rtp_vit_embedding_cache_inflight_qps |
| vit | 10367 | vit embedding cache tokens | 待埋点 | py_rtp_vit_embedding_cache_tokens |
| vit | 10624 | vit greennet success qps | 待埋点 | py_rtp_greennet_success_qps |
| vit | 10625 | vit greennet error qps | 待埋点 | py_rtp_greennet_error_qps |
| vit | 10626 | vit greennet rt | 待埋点 | py_rtp_greennet_rt |
| vit | 10627 | 提交cpu任务到预处理开始耗时 | 待埋点 | py_rtp_vit_preprocess_dispatch_rt |
| vit | 10628 | 预处理完成到主进程回调耗时 | 待埋点 | py_rtp_vit_preprocess_return_rt |
| vit | 10637 | 请求总耗时 | 待埋点 | py_rtp_vit_request_rt |
| master | 9102 | 前端路由异常：每秒多少次、什么原因 | 待埋点 | py_rtp_master_route_error_qps |
| master | 10524 | 前端路由成功：每秒多少次 | 待埋点 | py_rtp_master_route_qps |
| master | 9103 | 前端取得 Master 路由要多久：平均 / P99 | 待埋点 | py_rtp_master_route_rt |
| master | 9104 | Master 还有多少请求待调度 | 待埋点 | py_rtp_master_queue_length |
| master | 10289 | 缓存命中率：候选最高 / 选中估计 / Worker 实际 | 部分待埋点 | whale-lb.app.cache.routing.candidate.max.hit.tokens,whale-lb.app.cache.routing.selected.match.total.tokens,whale-lb.app.cache.routing.selected.match.hit.tokens |
| master | 10253 | 服务发现返回多少地址（不等于可接单机器） | 已上报 | whale-lb.app.engine.health.check.engine.worker.number.service.discovery.result |
| master | 10454 | 同步卡在哪个线程池：待执行任务数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10214 | 每台 Master 收到多少选机调用（非 ViT） | 已上报 | whale-lb.auto_tpm.request.count |
| master | 9004 | 取得路由慢在哪一段：各阶段平均耗时 | 已上报 | whale-lb.app.flexlb.dispatch.ack.time.ms,whale-lb.app.flexlb.route.submit.time.ms,whale-lb.app.routing.queue.wait.time.ms,whale-lb.app.flexlb.ack.to.response.time.ms,whale-lb.app.request.network.delay.ms,whale-lb.app.grpc.server.process.ms |
| master | 10219 | 请求到达 Master 前后耗时：平均 / 最大 | 已上报 | whale-lb.app.request.network.delay.ms,whale-lb.app.grpc.server.process.ms |
| master | 10221 | Master 从接入到选机完成：平均 / 最大 | 已上报 | whale-lb.app.flexlb.route.submit.time.ms |
| master | 10223 | 选机后凑批等了多久：平均 / 最大 | 已上报 | whale-lb.app.routing.queue.wait.time.ms |
| master | 10225 | 下发后多久收到 Worker 确认：平均 / 最大 | 已上报 | whale-lb.app.flexlb.dispatch.ack.time.ms |
| master | 10203 | 各 Worker 还有多少批次未完成对账 | 已上报 | whale-lb.app.flexlb.inflight.batch.count |
| master | 10206 | 最老在途记录已经多久未结算 | 已上报 | whale-lb.app.flexlb.inflight.max.age.ms |
| master | 10207 | 每秒清理多少条超 TTL 的在途记录 | 已上报 | whale-lb.app.flexlb.inflight.ttl.expired.qps |
| master | 10217 | 备 Master 转发到主 Master：结果 QPS | 已上报 | whale-lb.app.forward.to.master.result |
| master | 10234 | 每个派发批次包含多少输入 Tokens | 已上报 | whale-lb.app.engine.balancing.master.batch.total.tokens |
| master | 10235 | 为什么开始派发批次：各原因 QPS | 已上报 | whale-lb.app.engine.balancing.master.dispatch.reason |
| master | 10236 | 各派发原因下：平均每批多少请求 | 已上报 | whale-lb.app.engine.balancing.master.batch.size |
| master | 10243 | Prefill 批次执行多久：预测 / Worker 实际 | 已上报 | whale-lb.app.flexlb.batch.predicted.time.ms,whale-lb.app.flexlb.batch.actual.time.ms |
| master | 10244 | 批次执行预测偏差：实际减预测 | 已上报 | whale-lb.app.flexlb.batch.predict.gap.ms |
| master | 9101 | Worker 内部处理耗时：各阶段平均值 | 部分待埋点 | rtp_llm_model_forward_us,rtp_llm_rpc_remote_load_cache_start_rt_us,rtp_llm_rpc_poll_local_output_rt_us,rtp_llm_rpc_enqueue_request_rt_us,rtp_llm_rpc_get_rpc_connection_rt_us,rtp_llm_rpc_multimodal_process_rt_us,rtp_llm_rpc_remote_allocate_resource_rt_us |
| master | 10248 | ZK 当前记录哪台是主 Master | 已上报 | whale-lb.app.engine.zk.master.node |
| master | 10249 | 进程生命周期事件（当前指标名称待核实） | 待埋点 | whale-lb.graceful.lifecycle.event |
| master | 10250 | 主备选举事件标记（类型已汇总） | 已上报 | whale-lb.app.engine.zk.master.event |
| master | 10242 | 已选到各 Worker、但尚未派发的请求数 | 已上报 | whale-lb.app.flexlb.batcher.queue.size |
| master | 10260 | Worker 剩余工作预计还需多久（排队＋运行） | 已上报 | whale-lb.app.engine.health.check.running.queue.time |
| master | 10255 | 各角色 Worker 的 KV 有多满：平均 / 最满 | 已上报 | whale-lb.app.cache.used.kv.cache.ratio |
| master | 10239 | Decode 每台机器已确认＋已预留多少请求 | 已上报 | whale-lb.app.flexlb.decode.total.load |
| master | 10201 | 两套请求账本：Worker 已知 / Master 未结算 | 已上报 | whale-lb.app.engine.health.check.running.task.info.size,whale-lb.app.flexlb.inflight.request.count |
| master | 10261 | Master 调度账本还有多少请求未结算 | 已上报 | whale-lb.app.flexlb.scheduler.inflight.size |
| master | 10256 | 各 Worker 平均还有多少 KV Tokens：已用 / 可用 | 已上报 | whale-lb.app.cache.used.kv.cache.tokens,whale-lb.app.cache.available.kv.cache.tokens |
| master | 10240 | Decode 请求账本预计预留多少 KV Tokens | 已上报 | whale-lb.app.flexlb.decode.inflight.kv.reserved.tokens |
| master | 10241 | Decode 请求账本硬预留多少 KV Tokens | 已上报 | whale-lb.app.flexlb.decode.inflight.hard.kv.reserved.tokens |
| master | 10258 | 各角色单台 Worker 平均有多少 KV 总 Tokens | 已上报 | whale-lb.app.cache.total.kv.cache.tokens |
| master | 10280 | 工作是否集中在少数机器：剩余耗时方差 | 已上报 | whale-lb.app.engine.worker.info.running.query.len.var |
| master | 10281 | Decode 机器之间单步速度差多少：耗时方差 | 已上报 | whale-lb.app.engine.worker.info.step.latency.var |
| master | 10263 | 一次 Status 回包带回多少已完成请求 ID | 已上报 | whale-lb.app.engine.health.check.finished.task.list.size |
| master | 10264 | Worker 状态中报告多少缓存键 | 已上报 | whale-lb.app.cache.key.size |
| master | 10501 | 缓存复用率：执行实测 / 历史重复（Tokens 加权） | 部分待埋点 | whale-lb.app.cache.recent.key.hit.count,whale-lb.app.cache.recent.key.total.count,whale-lb.app.cache.theory.hit.count,whale-lb.app.cache.theory.total.count |
| master | 10452 | 选机为什么优先或放弃更多缓存：原因 QPS | 已上报 | whale-lb.app.cache.affinity.decision.qps |
| master | 10292 | 最终选机少用了多少已知缓存：损失比例 | 已上报 | whale-lb.app.cache.routing.candidate.max.hit.tokens,whale-lb.app.cache.routing.selected.match.hit.tokens |
| master | 10451 | 每秒输入与可复用 Tokens：候选最多 / 选中 | 已上报 | whale-lb.app.cache.routing.candidate.max.hit.tokens,whale-lb.app.cache.routing.selected.match.hit.tokens,whale-lb.app.cache.routing.selected.match.total.tokens |
| master | 10453 | Master 索引里登记了多少 Worker 地址 | 已上报 | whale-lb.app.cache.engine.views.map.size |
| master | 10304 | 每台 Master 索引包含多少不同缓存键 | 已上报 | whale-lb.app.cache.global.total.count |
| master | 10456 | 每台 Master 的缓存索引预计占多少内存 | 已上报 | whale-lb.app.cache.global.bytes |
| master | 10294 | Master 为每台 Worker 登记了多少缓存键 | 已上报 | whale-lb.app.cache.engine.local.count |
| master | 10291 | 每台 Worker 的索引记录预计占多少 Master 内存 | 已上报 | whale-lb.app.cache.engine.local.bytes |
| master | 10298 | 一次 Cache 更新新增 / 删除多少键 | 已上报 | whale-lb.app.cache.diff.added.blocks.size,whale-lb.app.cache.diff.removed.blocks.size |
| master | 10308 | 取得缓存更新锁后：更新索引要多久 | 已上报 | whale-lb.app.cache.update.engine.block.cache.rt |
| master | 10301 | 选机查询缓存索引要多久 | 已上报 | whale-lb.app.cache.find.matching.engines.rt |
| master | 10306 | 一个缓存块对应多少 Tokens | 已上报 | whale-lb.app.cache.block.size |
| master | 10312 | Master 启动累计：历史重复 / 输入 Tokens | 已上报 | whale-lb.app.cache.theory.hit.count,whale-lb.app.cache.theory.total.count |
| master | 10296 | Master 每秒做多少次缓存索引匹配 | 已上报 | whale-lb.app.cache.request.total |
| master | 10595 | FlexLB recent cache key hit/total count | 已上报 | whale-lb.app.cache.recent.key.hit.count,whale-lb.app.cache.recent.key.total.count |
| master | 10277 | Status 更新失败：原因与每秒次数 | 已上报 | whale-lb.app.engine.health.check.fail |
| master | 10525 | Status 回包在 Master 成功提交：QPS | 已上报 | whale-lb.app.engine.health.check.visitor.success.qps |
| master | 10299 | Cache 更新失败：原因与每秒次数 | 已上报 | whale-lb.app.cache.status.check.fail |
| master | 10526 | Cache 回包在 Master 成功处理：QPS | 已上报 | whale-lb.app.cache.status.check.visitor.success.qps |
| master | 10455 | 恢复成功时：距上次成功已经多久 | 已上报 | whale-lb.app.cache.status.check.success.period,whale-lb.app.engine.health.check.success.period |
| master | 10276 | Master 一轮同步多久完成：Status / Cache | 已上报 | whale-lb.app.engine.health.check.visitor.rt,whale-lb.app.cache.status.check.visitor.rt |
| master | 9201 | Worker 实际收到多少 Status / Cache 查询 | 待埋点 | rtp_llm_rpc_worker_status_qps,rtp_llm_rpc_cache_status_qps |
| master | 9202 | Worker 服务端处理查询多久：平均 / 最大 | 待埋点 | rtp_llm_rpc_worker_status_total_rt_us,rtp_llm_rpc_cache_status_total_rt_us |
| master | 10446 | Status RPC 发起线程池：执行 / 排队 / 线程数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10447 | Status 回包本地提交线程池：执行 / 排队 / 线程数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10448 | Cache 轮询与处理线程池：执行 / 排队 / 线程数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10445 | 周期轮询调度线程池：执行 / 排队 / 线程数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10325 | Client gRPC 回调线程池：执行 / 排队 / 线程数 | 已上报 | whale-lb.app.engine.balancing.thread.pool.info |
| master | 10326 | Client gRPC EventLoop：线程数 / 待处理任务 | 已上报 | whale-lb.app.engine.balancing.event.loop.group.info |
| master | 10329 | Client RPC 成功耗时：平均 / 最大 | 已上报 | whale-lb.app.grpc.call.duration |
| master | 10331 | Client RPC 成功 QPS：按操作 / 是否重试 | 已上报 | whale-lb.app.grpc.call.count |
| master | 10330 | Client 回调交付延迟：平均 / 最大 | 待埋点 | whale-lb.app.grpc.delivery.delay_ms |
| master | 10332 | Client 重连前：旧 Channel 已存活多久 | 已上报 | whale-lb.app.grpc.connection.duration |
| master | 10327 | 每台 Master 连接池有多少 gRPC Channel | 已上报 | whale-lb.app.grpc.channel.pool.size |
| master | 10209 | Worker Prefill 线程池：正在执行任务的线程 | 待埋点 | rtp_llm_prefill_pool_active |
| master | 10210 | Worker Prefill 线程池：尚未执行的任务 | 待埋点 | rtp_llm_prefill_pool_queued |
| master | 10211 | Worker Prefill 线程池：活跃线程 / 配置上限 | 待埋点 | rtp_llm_prefill_pool_active,rtp_llm_prefill_pool_thread_max |
| master | 10212 | Worker Prefill 拒绝指标（计数口径待核实） | 待埋点 | rtp_llm_prefill_pool_rejected |
| master | 10611 | Client 连接异常：传输事件 / 底层原因 | 待埋点 | whale-lb.app.grpc.call.failure |
| master | 10608 | Client RPC 失败 QPS：原因 / gRPC 状态 | 待埋点 | whale-lb.app.grpc.call.failure |
| master | 10609 | Client RPC 失败 QPS：已确认本地 / 原因未确认 | 待埋点 | whale-lb.app.grpc.call.failure |
| master | 10610 | Client RPC 失败位置：本地阶段 / 传输进度 | 待埋点 | whale-lb.app.grpc.call.failure |
| master | 10612 | Status/Cache 更新失败：操作 / 阶段 / 原因 | 待埋点 | whale-lb.app.status.failure.detail |
| master | 10613 | Status/Cache 更新失败：归因 / gRPC 状态 | 待埋点 | whale-lb.app.status.failure.detail |
| master | 10176 | 每台 Master 各优先级收到多少选机调用 | 已上报 | whale-lb.auto_tpm.request.count |
| master | 10180 | 各优先级从接入到调度返回多久 | 已上报 | whale-lb.auto_tpm.schedule.latency_ms |
| master | 10186 | 待核实：请求调度 SLO 指标（本版未找到上报） | 待埋点 | whale-lb.auto_tpm.request.slo_ms |
| master | 10195 | 待核实：Deadline 未达标指标（本版未找到上报） | 待埋点 | whale-lb.auto_tpm.deadline_miss.count |
| master | 10182 | 待核实：常规放置指标（本版未找到上报） | 待埋点 | whale-lb.auto_tpm.normal_placement.count |
| master | 10181 | Prefill 已选机后：每台平均积压多少请求 | 已上报 | whale-lb.auto_tpm.prefill.queue_depth |
| master | 10177 | 抢占计划：检查 / 提交尝试 / 旧冲突指标 | 部分待埋点 | whale-lb.auto_tpm.eviction_plan.count,whale-lb.auto_tpm.eviction_commit.count,whale-lb.auto_tpm.plan_conflict.count |
| master | 10179 | 每秒选中多少个被抢占请求 | 已上报 | whale-lb.auto_tpm.victim.count |
| master | 10187 | 每秒记录多少次优先级抢占 | 已上报 | whale-lb.auto_tpm.priority_preempt.count |
| master | 10194 | 取消推进到哪一步：产生意图 / 发 RPC / 确认 | 已上报 | whale-lb.auto_tpm.cancel.qps,whale-lb.auto_tpm.cancel.request.count,whale-lb.auto_tpm.cancel.confirm.count |
| master | 10527 | 取消等待窗口内未等到确认：QPS | 已上报 | whale-lb.auto_tpm.cancel.timeout.count |
| master | 10199 | 待核实：抢占计划年龄（本版未找到上报） | 待埋点 | whale-lb.auto_tpm.plan_age_ms |
| master | 10196 | 抢占结束对账时：找不到在途记录的次数 | 已上报 | whale-lb.auto_tpm.inflight_settle_miss.count |
| master | 10191 | Decode 账本：正在运行 / 尚未接收 / 已接收待运行 | 已上报 | whale-lb.auto_tpm.decode.running.count,whale-lb.auto_tpm.decode.reserved.count,whale-lb.auto_tpm.decode.accepted.count |
| master | 10190 | Decode 已派给引擎多少请求（账本计数） | 已上报 | whale-lb.auto_tpm.decode.engine_load |
| master | 10184 | Decode 尚未下发请求的硬 KV 预留 | 已上报 | whale-lb.auto_tpm.decode.shadow_kv_reserved |
| master | 10185 | 每次抢占释放多少账本 KV 预留 | 已上报 | whale-lb.auto_tpm.victim.kv_tokens |
| master | 10340 | Master JVM 堆内存：当前已用 / 配置上限 | 已上报 | jvm_mem_heap_max,jvm_mem_heap_used |
| master | 10342 | Master JVM BufferPool：已用 / 当前容量 | 已上报 | jvm_buffer_pool_mapped_capacity,jvm_buffer_pool_mapped_used,jvm_buffer_pool_direct_capacity,jvm_buffer_pool_direct_used |
| master | 10338 | GC 用时采集值：Full / Young（口径待核实） | 待埋点 | jvm_gc_fgc_time,jvm_gc_ygc_time |
| master | 10339 | GC 次数采集值：Full / Young（口径待核实） | 待埋点 | jvm_gc_fgc_count,jvm_gc_ygc_count |
| master | 10334 | 容器 CPU 采集值（比例口径待核实） | 待埋点 | pod_cpu_user_limit_percent |
| master | 10335 | 容器最近 1 分钟 CPU Load | 待埋点 | pod_cpu_load_average_1m_v2 |
| master | 10336 | 容器内存 v3 采集值（比例口径待核实） | 待埋点 | pod_memory_util_v3 |
| master | 10337 | 容器内存旧采集值（比例口径待核实） | 待埋点 | pod_memory_util |
| master | 10343 | Master JVM 线程：总数 / 可运行 / 死锁 | 部分待埋点 | jvm_thread_count,jvm_thread_deadlock_count,jvm_thread_runnable_count |
| master | 10345 | 容器 OOM 事件采集值（累计口径待核实） | 待埋点 | pod_event_oom |
| master | 10344 | 容器每秒发生多少次 Major Page Fault | 待埋点 | pod_memory_pgmajfault_ps |
| master | 10341 | 各 Master TCP 连接处在哪个状态 | 待埋点 | pod_network_tcp_state |
| master | 10435 | Master JVM 非堆内存：当前已用 / 上限 | 已上报 | jvm_mem_non_heap_max,jvm_mem_non_heap_used |
| master | 10436 | JVM 编译代码缓存各区域（比例口径待核实） | 待埋点 | jvm_mem_pools_code_cache_usage,jvm_mem_pools_codeheap_non_profiled_nmethods_usage,jvm_mem_pools_codeheap_profiled_nmethods_usage,jvm_mem_pools_codeheap_non_nmethods_usage |
| master | 10605 | Netty Direct：被缓冲区占用的内存（pinned） | 待埋点 | whale-lb.app.netty.direct.pinned.bytes |
| master | 10604 | Netty Direct：分配器保留内存（reserved） | 待埋点 | whale-lb.app.netty.direct.reserved.bytes |
| master | 10606 | Netty 内存采样耗时 | 待埋点 | whale-lb.app.netty.memory.sample.duration.ms |
| master | 10607 | Netty 内存采样：两次成功采样间隔 | 待埋点 | whale-lb.app.netty.memory.sample.interval.ms |
| prefill | 104 | cpu / percent | 待埋点 | pod_cpu_limit_usage |
| prefill | 10122 | tcp 联接数 | 待埋点 | pod_network_tcp_state |
| prefill | 92 | oom | 待埋点 | pod_event_oom |
| prefill | 91 | core dump | 待埋点 | pod_event_coredump_existence |
| prefill | 20 | 容器gpu利用率 | 待埋点 | container_gpu_duty_cycle |
| prefill | 17 | GPU Memory Usage | 待埋点 | pod_gpu_memory_used_bytes |
| prefill | 455 | Major Page Fault 次数 / Sec | 待埋点 | pod_memory_pgmajfault_ps |
| prefill | 117 | memory  rss util | 待埋点 | pod_memory_util |
| prefill | 266 | Gpu卡数 | 待埋点 |  |
| prefill | 453 | pod_memory_util(by cgroup) | 待埋点 | pod_memory_util_v3 |
| prefill | 208 | pod GPU利用率 | 待埋点 | pod_gpu_duty_cycle |
| prefill | 93 | pod mem used | 待埋点 | pod_memory_usage_bytes_v2,pod_memory_usage_bytes,pod_memory_cache_bytes |
| prefill | 181 | AMD GPU Util | 待埋点 |  |
| prefill | 401 | Pod docker内存利用率(memory) | 待埋点 | pod_memory_util |
| prefill | 210 | 因为相邻Pod导致的调度排队时间占比 | 待埋点 | pod_cpu_queue_sibling_ps,pod_cpu_queue_other_ps,pod_cpu_serve_time_ps |
| prefill | 277 | tensor core利用率 avg | 待埋点 | pod_gpu_pipe_tensor_active |
| prefill | 278 | tensorflops used avg | 待埋点 | pod_gpu_tensortflops_used |
| prefill | 275 | tensor core利用率 max | 待埋点 | pod_gpu_pipe_tensor_active |
| prefill | 269 | pod SM利用率 | 待埋点 | pod_gpu_sm_util |
| prefill | 271 | pod SM occupancy | 待埋点 | pod_gpu_sm_occupancy |
| prefill | 273 | tensorflops used max | 待埋点 | pod_gpu_tensortflops_used |
