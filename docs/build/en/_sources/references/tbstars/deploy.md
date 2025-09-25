# tbstars-moe-42 42B-A3.5B-MoE

# Recommended Deployment

It is recommended to use a combination of PD (Prefill-Decode) separation and DP (Data Parallelism) for optimal performance. PD separation is employed because of the MHA (Multi-Head Attention) structure, which results in substantial GPU memory consumption from KV cache. Without PD separation, the system would face KV cache insufficiency when handling large max sequence lengths. DP is implemented for similar reasons, aiming to reduce the GPU memory footprint of model weights on each GPU card.

For deployment, it is recommended to enable DeepEP, FP8 quantization, and FP8 KV cache.

* Framework version: v0.2.0

# Advanced Configuration for Cluster Deployment

## H20 96G

### Prefill Node Specifications

Single-node resource requirements: 2 GPUs per node
| CPU Cores | Memory | GPU Card | GPU Count | Disk Space | RDMA NICs |
|----------|--------|----------|-----------|------------|-----------|
| 96       | 250GB  | H20      | 2         | 180GB      | 2         |

### Prefill Advanced Arguments

```
LD_PRELOAD=/usr/lib64/libjemalloc.so
ali_extend_devs=/dev/gdrdrv:/dev/gdrdrv:rwm \
NVSHMEM_IBGDA_NUM_RC_PER_PE=12 \
NVSHMEM_DISABLE_P2P=1 \
NVSHMEM_IB_ENABLE_IBGDA=1 \
NVSHMEM_IBGDA_NIC_HANDLER=gpu \
NVSHMEM_QP_DEPTH=1024 \
NVSHMEM_CUMEM_GRANULARITY=536870912  \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_IB_TRAFFIC_CLASS=160 \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_DISABLE_ABORT=1 \
NCCL_IB_GID_INDEX=3 \
NCCL_IB_SL=5 \
NCCL_IB_TC=160 \
NCCL_IB_HCA=mlx5 \
ACCL_RX_DEPTH=32 \
ACCL_TX_DEPTH=512 \
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
--model_type mixtbstars \
--act_type bf16 \
--max_seq_len 131072 \
--max_batch_size 512 \
--concurrency_limit 512 \
--think_mode 0 \
--tp_size 1 \
--dp_size 2 \
--world_rank 0 \
--local_world_size 2 \
--reuse_cache 0 \
--cache_store_rdma_mode 1 \
--cache_store_rdma_connect_timeout_ms 800 \
--balance_method mix \
--load_balance 1 \
--use_deepep_internode 0 \
--use_deepep_low_latency 0 \
--use_deepep_moe 1 \
--enable_layer_micro_batch 0 \
--seq_size_per_block 64 \
--log_level INFO \
--reserver_runtime_mem_mb 42000 \
--device_reserve_memory_bytes -4096000000 \
--start_port 8090 \
--warm_up 0 \
--quantization FP8_PER_BLOCK \
--fp8_kv_cache 1
```

### Decode Node Specifications

Single-node resource requirements: 4 GPUs per node
| CPU Cores | Memory | GPU Card | GPU Count | Disk Space | RDMA NICs |
|----------|--------|----------|-----------|------------|-----------|
| 96       | 250GB  | H20      | 4         | 180GB      | 4         |

### Decode Advanced Arguments

```
LD_PRELOAD=/usr/lib64/libjemalloc.so
ali_extend_devs=/dev/gdrdrv:/dev/gdrdrv:rwm \
NVSHMEM_IBGDA_NUM_RC_PER_PE=12 \
NVSHMEM_DISABLE_P2P=1 \
NVSHMEM_IB_ENABLE_IBGDA=1 \
NVSHMEM_IBGDA_NIC_HANDLER=gpu \
NVSHMEM_QP_DEPTH=1024 \
NVSHMEM_CUMEM_GRANULARITY=536870912  \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_IB_TRAFFIC_CLASS=160 \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_DISABLE_ABORT=1 \
NCCL_IB_GID_INDEX=3 \
NCCL_IB_SL=5 \
NCCL_IB_TC=160 \
NCCL_IB_HCA=mlx5 \
ACCL_RX_DEPTH=32 \
ACCL_TX_DEPTH=512 \
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
/opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path XXXX \
--model_type mixtbstars \
--act_type bf16 \
--max_seq_len 131072 \
--max_batch_size 128 \
--concurrency_limit 128 \
--think_mode 0 \
--tp_size 1 \
--dp_size 4 \
--world_rank 0 \
--local_world_size 4 \
--reuse_cache 0 \
--cache_store_rdma_mode 1 \ # delete
--cache_store_rdma_connect_timeout_ms 800 \
--balance_method mix \
--load_balance 1 \
--use_deepep_internode 0 \
--use_deepep_low_latency 1 \
--use_deepep_moe 1 \
--original_checkpoint_path XXXX \
--enable_layer_micro_batch 0 \
--seq_size_per_block 64 \
--log_level INFO \
--reserver_runtime_mem_mb 2048 \
--device_reserve_memory_bytes -3072000000 \
--start_port 8090 \
--warm_up 0 \
--quantization FP8_PER_BLOCK \
--fp8_kv_cache 1
```