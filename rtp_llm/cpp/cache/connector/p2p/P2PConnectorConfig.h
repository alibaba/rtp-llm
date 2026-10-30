#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rtp_llm {

inline int64_t getP2PTransferListenPort(int64_t cache_store_listen_port) {
    // Reuse the reserved per-worker slot right after cache_store_listen_port.
    // Keep port 0 unchanged so tests can still request an ephemeral port.
    return cache_store_listen_port > 0 ? cache_store_listen_port + 1 : cache_store_listen_port;
}

struct P2PConnectorSchedulerConfig {
    int64_t load_cache_timeout_ms = 5000;
    std::vector<std::string> worker_grpc_addrs;
    std::vector<std::string> worker_addrs;
    std::string                          pipeline_layout;
    int64_t                  p2p_lease_query_timeout_ms                   = 500;
    int64_t                  p2p_cancel_broadcast_timeout_ms              = 1000;
    int64_t                     p2p_cancelled_keys_ttl_ms    = 3600 * 1000;
    std::shared_ptr<const CacheTopology> topology;
    bool                                 use_group_cache_transfer_policy = false;
    bool                                 use_mla                         = false;
    bool                                 local_full_page_rr              = false;
    int                                  decode_transfer_layer_end       = std::numeric_limits<int>::max();
    std::map<std::string, size_t>        physical_block_bytes_by_group;
    size_t                               cache_key_tokens_per_block = 1;
    int                                  cp_rank = 0;
    int                                  cp_size = 1;

    // TP-only planning mirrors the peer layout. PP planning uses the actual
    // stage descriptions carried by GetPeerInfo and StartLoad.
    ParallelismConfig parallelism_config;
    RoleType          role_type = RoleType::PDFUSION;

    static P2PConnectorSchedulerConfig create(const RuntimeConfig&     runtime_config,
                                              const CacheStoreConfig&  cache_store_config,
                                              const ParallelismConfig& parallelism_config,
                                              const PDSepConfig&       pd_sep_config) {
        P2PConnectorSchedulerConfig config;
        config.parallelism_config                     = parallelism_config;
        config.role_type                              = pd_sep_config.role_type;
        config.load_cache_timeout_ms                  = pd_sep_config.load_cache_timeout_ms;
        config.worker_grpc_addrs                      = runtime_config.worker_grpc_addrs;
        config.worker_addrs                           = runtime_config.worker_addrs;
        config.p2p_lease_query_timeout_ms = cache_store_config.p2p_lease_query_timeout_ms;
        config.p2p_cancel_broadcast_timeout_ms = cache_store_config.p2p_cancel_broadcast_timeout_ms;
        config.p2p_cancelled_keys_ttl_ms       = cache_store_config.p2p_cancelled_keys_ttl_ms;
        return config;
    }
};

struct P2PConnectorWorkerConfig {
    transfer::TransferBackendConfig transfer_backend_config;

    int p2p_prefill_sender_thread_count = 4;
    int p2p_prefill_sender_queue_size   = 10000;

    int64_t p2p_cancelled_keys_ttl_ms                = 3600 * 1000;
    int64_t load_cache_timeout_ms = 5000;

    int64_t  tp_size       = 1;
    int64_t  tp_rank       = 0;
    uint32_t layer_all_num = 0;
    uint32_t global_layer_begin = 0;

    std::shared_ptr<const CacheTopology> topology;
    size_t                               cache_key_tokens_per_block = 1;
    int                                  cp_size = 1;

    // Only LINEAR_QKV needs the physical spec to build byte ranges. A tag may
    // be shared by the main model and a draft with different local geometry.
    std::vector<std::map<std::string, std::shared_ptr<const KVCacheSpec>>> linear_specs_by_layer;

    const KVCacheSpec* linearSpecForLayer(size_t layer_id, const std::string& tag) const {
        return linear_specs_by_layer[layer_id].at(tag).get();
    }

    static P2PConnectorWorkerConfig create(const CacheStoreConfig&  cache_store_config,
                                           const PDSepConfig&       pd_sep_config,
                                           const ParallelismConfig& parallelism_config,
                                           uint32_t                 layer_all_num,
                                           size_t                   llm_kv_block_size_bytes) {
        P2PConnectorWorkerConfig config;
        config.load_cache_timeout_ms = pd_sep_config.load_cache_timeout_ms;
        config.p2p_prefill_sender_thread_count = cache_store_config.p2p_prefill_sender_thread_count;
        config.p2p_prefill_sender_queue_size   = cache_store_config.p2p_prefill_sender_queue_size;
        config.transfer_backend_config.cache_store_rdma_mode         = pd_sep_config.cache_store_rdma_mode;
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
        config.transfer_backend_config.p2p_rdma_enable_h2d_copy = cache_store_config.p2p_rdma_enable_h2d_copy;
        config.transfer_backend_config.p2p_rdma_staging_total_bytes =
            cache_store_config.p2p_rdma_staging_total_bytes;
        config.transfer_backend_config.llm_kv_block_size_bytes = llm_kv_block_size_bytes;
        config.transfer_backend_config.rdma_disconnect_after_deadline_ms =
            cache_store_config.p2p_transfer_not_done_resource_hold_ms;
        config.p2p_cancelled_keys_ttl_ms                = cache_store_config.p2p_cancelled_keys_ttl_ms;
        config.tp_size                                 = parallelism_config.tp_size;
        config.tp_rank                                 = parallelism_config.tp_rank;
        config.layer_all_num                           = layer_all_num;
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
                                     size_t                   llm_kv_block_size_bytes) {
        P2PConnectorConfig config;
        config.role_type = pd_sep_config.role_type;
        config.tp_rank   = parallelism_config.tp_rank;
        config.scheduler_config =
            P2PConnectorSchedulerConfig::create(runtime_config, cache_store_config, parallelism_config, pd_sep_config);
        config.worker_config = P2PConnectorWorkerConfig::create(
            cache_store_config, pd_sep_config, parallelism_config, layer_all_num, llm_kv_block_size_bytes);
        // Apply the same bounded fallback as the scheduler before init validates
        // the production connector, including Prefill's worker configuration.
        const auto    timeout                         = pd_sep_config.load_cache_timeout_ms;
        const int64_t normalized_timeout              = timeout > 0 ?
                                                            std::min<int64_t>(timeout, std::numeric_limits<int32_t>::max()) :
                                                            PDSepConfig{}.load_cache_timeout_ms;
        config.scheduler_config.load_cache_timeout_ms = normalized_timeout;
        config.worker_config.load_cache_timeout_ms    = normalized_timeout;
        return config;
    }
};

}  // namespace rtp_llm
