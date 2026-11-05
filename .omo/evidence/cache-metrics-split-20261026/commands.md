## cp-launch-6e6ab2ce.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache build //rtp_llm:rtp_llm --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_cp_6e6ab2ce_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_cp_6e6ab2ce_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_cp_6e6ab2ce_20261026.signal

```

## tail-launch-d117337a.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache build //rtp_llm:rtp_llm --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_tail_d117337a_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_tail_d117337a_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_tail_d117337a_20261026.signal

```

## prefix-launch-8f842c28.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache build //rtp_llm:rtp_llm --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_prefix_8f842c28_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_prefix_8f842c28_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_prefix_8f842c28_20261026.signal

```

## optional-launch.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache build //rtp_llm:rtp_llm --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_optional_00bb35ca_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_optional_00bb35ca_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_optional_00bb35ca_20261026.signal

```

## regressions-launch-b3b6ea49.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache test //rtp_llm/cpp/cache/block_tree_cache/test:block_tree_test //rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test //rtp_llm/cpp/cache/block_tree_cache/load/test:block_tree_loader_test //rtp_llm/cpp/cache/block_tree_cache/load/test:load_async_context_test //rtp_llm/cpp/cache/block_tree_cache/store/test:block_tree_storer_test //rtp_llm/cpp/cache/block_tree_cache/store/test:store_task_runner_test //rtp_llm/cpp/cache/block_tree_cache/evict/test:block_tree_evictor_test //rtp_llm/cpp/cache/block_tree_cache/storage_backend/test:storage_backend_test //rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:kvcm_mock_full_linear_test //rtp_llm/cpp/engine_base/stream/test:stream_cache_resource_test //rtp_llm/cpp/cache/test:single_type_kv_cache_allocator_test //rtp_llm/cpp/cache/test:hybrid_kv_cache_allocator_cp_shard_test //rtp_llm/cpp/cache/test:prefill_cache_hit_metrics_contract_test //rtp_llm/cpp/cache/test:kv_cache_layout_view_test //rtp_llm/cpp/cache/test:kv_cache_manager_cp_test //rtp_llm/cpp/normal_engine/test:engine_test //rtp_llm/cpp/normal_engine/test:tail_reuse_cp_completion_integration_test //rtp_llm/cpp/normal_engine/test:tail_reuse_tp_subgroups_integration_test --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --nocache_test_results --test_output=errors --test_timeout=3000 --keep_going --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_nometrics_regressions_b3b6ea49_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_nometrics_regressions_b3b6ea49_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_nometrics_regressions_b3b6ea49_20261026.signal

```

## block-tree-launch-8f842c28.log

```text
Launching: bazelisk --batch --output_user_root=/data0/songchenyi.scy/.cache/bazel_cuda13_RTP-LLM-20261024-Partial_Match_reorg_cache test //rtp_llm/cpp/cache/block_tree_cache/test:block_tree_test --config=cuda13 --jobs=200 --disk_cache=/home/songchenyi.scy/.cache/disk_cache --config=daily_aone_bazel_remote --remote_header=x-aone-bazel-api-key=ai-infra-cicd --nocache_test_results --test_output=errors --test_timeout=3000 --keep_going --config=daily_aone_bazel_cache
Log: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_block_tree_8f842c28_20261026.log
Signal: /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_block_tree_8f842c28_20261026.signal
Started (detached; result unverified). Poll with: ./poll_signal.sh /data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/build_logs/split_block_tree_8f842c28_20261026.signal

```
