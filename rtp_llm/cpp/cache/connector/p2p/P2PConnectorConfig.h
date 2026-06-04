#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {

inline int64_t getP2PTransferListenPort(int64_t cache_store_listen_port) {
    // Reuse the reserved per-worker slot right after cache_store_listen_port.
    // Keep port 0 unchanged so tests can still request an ephemeral port.
    return cache_store_listen_port > 0 ? cache_store_listen_port + 1 : cache_store_listen_port;
}

struct P2PConnectorSchedulerConfig {
    std::vector<std::string> worker_grpc_addrs;
    std::vector<std::string> worker_addrs;
    std::vector<std::string> p2p_worker_addrs;
    int64_t                  p2p_transfer_not_done_resource_hold_ms       = 10 * 1000;
    int                      p2p_resource_store_timeout_check_interval_ms = 100;
    int64_t                  p2p_cancel_broadcast_timeout_ms              = 1000;
    int64_t                  p2p_prefill_resource_hold_ms                 = 300 * 1000;
    // Clamp for the P2P RPC deadline at decode/prefill entry; see
    // CacheStoreConfig::p2p_max_transfer_deadline_ms for rationale.
    int64_t                     p2p_max_transfer_deadline_ms = 300 * 1000;
    int64_t                     p2p_cancelled_keys_ttl_ms    = 3600 * 1000;
    std::vector<CacheGroupType> layer_attn_types;

    static P2PConnectorSchedulerConfig create(const RuntimeConfig&    runtime_config,
                                              const CacheStoreConfig& cache_store_config,
                                              const PDSepConfig&      pd_sep_config) {
        P2PConnectorSchedulerConfig config;
        config.worker_grpc_addrs                      = runtime_config.worker_grpc_addrs;
        config.worker_addrs                           = runtime_config.worker_addrs;
        config.p2p_worker_addrs                       = runtime_config.p2p_worker_addrs;
        config.p2p_transfer_not_done_resource_hold_ms = cache_store_config.p2p_transfer_not_done_resource_hold_ms;
        config.p2p_resource_store_timeout_check_interval_ms =
            cache_store_config.p2p_resource_store_timeout_check_interval_ms;
        config.p2p_cancel_broadcast_timeout_ms = cache_store_config.p2p_cancel_broadcast_timeout_ms;
        config.p2p_prefill_resource_hold_ms    = cache_store_config.p2p_prefill_resource_hold_ms;
        config.p2p_max_transfer_deadline_ms    = cache_store_config.p2p_max_transfer_deadline_ms;
        config.p2p_cancelled_keys_ttl_ms       = cache_store_config.p2p_cancelled_keys_ttl_ms;
        return config;
    }
};

struct P2PConnectorWorkerConfig {
    transfer::TransferBackendConfig transfer_backend_config;

    int64_t p2p_read_steal_before_deadline_ms       = 250;
    int64_t p2p_read_return_before_deadline_ms      = 100;
    int64_t p2p_layer_cache_buffer_store_timeout_ms = 100 * 1000;

    int64_t  tp_size       = 1;
    int64_t  tp_rank       = 0;
    uint32_t layer_all_num = 0;
    bool     is_mla        = false;

    static P2PConnectorWorkerConfig create(const CacheStoreConfig&  cache_store_config,
                                           const PDSepConfig&       pd_sep_config,
                                           const ParallelismConfig& parallelism_config,
                                           uint32_t                 layer_all_num,
                                           bool                     is_mla = false) {
        P2PConnectorWorkerConfig config;
        config.transfer_backend_config.cache_store_rdma_mode         = pd_sep_config.cache_store_rdma_mode;
        config.transfer_backend_config.rdma_transfer_wait_timeout_ms = cache_store_config.rdma_transfer_wait_timeout_ms;
        config.transfer_backend_config.messager_io_thread_count      = cache_store_config.messager_io_thread_count;
        config.transfer_backend_config.messager_worker_thread_count  = cache_store_config.messager_worker_thread_count;
        config.transfer_backend_config.rdma_max_block_pairs_per_connection =
            cache_store_config.rdma_max_block_pairs_per_connection;
        config.transfer_backend_config.cache_store_listen_port =
            getP2PTransferListenPort(pd_sep_config.cache_store_listen_port);
        config.transfer_backend_config.cache_store_tcp_anet_rpc_thread_num =
            cache_store_config.cache_store_tcp_anet_rpc_thread_num;
        config.transfer_backend_config.cache_store_tcp_anet_rpc_queue_num =
            cache_store_config.cache_store_tcp_anet_rpc_queue_num;
        config.transfer_backend_config.cache_store_tcp_worker_queue_size =
            cache_store_config.cache_store_tcp_worker_queue_size;
        config.transfer_backend_config.rdma_transfer_worker_thread_count =
            cache_store_config.rdma_transfer_worker_thread_count;
        config.transfer_backend_config.rdma_transfer_worker_queue_size =
            cache_store_config.rdma_transfer_worker_queue_size;
        config.p2p_layer_cache_buffer_store_timeout_ms = cache_store_config.p2p_layer_cache_buffer_store_timeout_ms;
        config.p2p_read_steal_before_deadline_ms       = cache_store_config.p2p_read_steal_before_deadline_ms;
        config.p2p_read_return_before_deadline_ms      = cache_store_config.p2p_read_return_before_deadline_ms;
        config.tp_size                                 = parallelism_config.tp_size;
        config.tp_rank                                 = parallelism_config.tp_rank;
        config.layer_all_num                           = layer_all_num;
        config.is_mla                                  = is_mla;
        return config;
    }
};

struct P2PConnectorConfig {
    RoleType role_type = RoleType::PDFUSION;
    int      tp_rank   = 0;

    P2PConnectorSchedulerConfig scheduler_config;
    P2PConnectorWorkerConfig    worker_config;

    static P2PConnectorConfig create(const RuntimeConfig&     runtime_config,
                                     const CacheStoreConfig&  cache_store_config,
                                     const ParallelismConfig& parallelism_config,
                                     const PDSepConfig&       pd_sep_config,
                                     uint32_t                 layer_all_num,
                                     bool                     is_mla = false) {
        P2PConnectorConfig config;
        config.role_type = pd_sep_config.role_type;
        config.tp_rank   = parallelism_config.tp_rank;
        config.scheduler_config =
            P2PConnectorSchedulerConfig::create(runtime_config, cache_store_config, pd_sep_config);
        config.worker_config = P2PConnectorWorkerConfig::create(
            cache_store_config, pd_sep_config, parallelism_config, layer_all_num, is_mla);
        return config;
    }
};

}  // namespace rtp_llm
