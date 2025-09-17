# KiMi-K2 Recommended Deployment

It is recommended to use PD (Prefill-Decode) separation for optimal performance.

* Framework version: v0.2.0

# Advanced Configuration

The resource requirements and startup parameters for prefill and decode are described separately below.

## Prefill

### Resource Requirements

Single-node resource specifications requiring 4 nodes for multi-node deployment:
| CPU Cores | Memory | GPU Cards | GPU Count | Disk Space | RDMA NICs |
|----------|--------|-----------|-----------|------------|-----------|
| 96       | 960GB  | H20 | 8         | 180GB      | 8         |

### Advanced Arguments

```
LD_PRELOAD=/usr/lib64/libjemalloc.so
ali_extend_devs=/dev/gdrdrv:/dev/gdrdrv:rwm \
RDMA_CONNECT_RETRY_TIMES=2 \
LOAD_BALANCE_POLICY_NAME=WRR \
MAX_RPC_TIMEOUT_MS=1800000 \
PD_SEPARATION=1 \
ROLE_TYPE=PREFILL \
PREFILL_MAX_WAIT_TIMEOUT_US=180000000 \
PREFILL_RETRY_TIMEOUT_MS=20 \
PREFILL_RETRY_TIME=1 \
USE_CACHE_STORE=1 \
LOAD_CACHE_TIMEOUT_MS=180000 \
SYNC_STATUS_INTERVAL_MS=200 \
ENABLE_MERGE_W13=1 \
/opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path XXXX \
--model_type deepseek3 \
--act_type bf16 \
--sp_model_type deepseek-v3-mtp \
--sp_checkpoint_path XXXXXXX \
--sp_act_type bf16 \
--sp_type mtp \
--sp_min_token_match 2 \
--sp_max_token_match 2 \
--max_seq_len 32768 \
--max_batch_size 1024 \
--concurrency_limit 1024 \
--think_mode 1 \
--tp_size 1 \
--dp_size 32 \
--world_rank 0 \
--local_world_size 8 \
--reuse_cache 0 \
--cache_store_rdma_mode 1 \
--cache_store_rdma_connect_timeout_ms 800 \
--balance_method mix \
--load_balance 1 \
--use_deepep_internode 1 \
--use_deepep_low_latency 0 \
--use_deepep_moe 1 \
--enable_layer_micro_batch 1 \
--seq_size_per_block 64 \
--log_level INFO \
--reserver_runtime_mem_mb 76800 \
--device_reserve_memory_bytes -12800000000 \
--start_port 8090 \
--warm_up 0
```

## Decode

### Resource Requirements

Single-node resource specifications requiring 27 nodes for multi-node deployment:
| CPU Cores | Memory | GPU Cards | GPU Count | Disk Space | RDMA NICs |
|----------|--------|-----------|-----------|------------|-----------|
| 96       | 250GB  | H20 | 8         | 180GB      | 8         |

### Advanced Arguments

```
LD_PRELOAD=/usr/lib64/libjemalloc.so
ali_extend_devs=/dev/gdrdrv:/dev/gdrdrv:rwm \
RDMA_CONNECT_RETRY_TIMES=2 \
MAX_RPC_TIMEOUT_MS=1800000 \
PD_SEPARATION=0 \
ROLE_TYPE=DECODE \
PREFILL_MAX_WAIT_TIMEOUT_US=180000000 \
PREFILL_RETRY_TIMEOUT_MS=20 \
PREFILL_RETRY_TIME=1 \
USE_CACHE_STORE=1 \
LOAD_CACHE_TIMEOUT_MS=180000 \
SYNC_STATUS_INTERVAL_MS=200 \
ENABLE_MERGE_W13=1 \
REDUNDANT_EXPERT=48 \
/opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path XXXX \
--model_type deepseek3 \
--act_type bf16 \
--sp_model_type deepseek-v3-mtp \
--sp_checkpoint_path XXXXXXX \
--sp_act_type bf16 \
--sp_type mtp \
--sp_min_token_match 2 \
--sp_max_token_match 2 \
--max_seq_len 32768 \
--max_batch_size 1024 \
--concurrency_limit 1024 \
--think_mode 1 \
--tp_size 1 \
--dp_size 144 \
--world_rank 0 \
--local_world_size 8 \
--reuse_cache 0 \
--cache_store_rdma_mode 1 \ # delete
--cache_store_rdma_connect_timeout_ms 800 \
--balance_method mix \
--load_balance 1 \
--use_deepep_internode 0 \
--use_deepep_low_latency 1 \
--use_deepep_moe 1 \
--original_checkpoint_path XXXX \
--eplb_stats_window_size 10 \
--eplb_update_time 20 \
--eplb_mode EPLB \
--eplb_force_repack 1 \
--redundant_expert 32 \
--enable_layer_micro_batch 2 \
--seq_size_per_block 64 \
--log_level INFO \
--reserver_runtime_mem_mb 6144 \
--device_reserve_memory_bytes -6144000000 \
--start_port 8090 \
--warm_up 0
```