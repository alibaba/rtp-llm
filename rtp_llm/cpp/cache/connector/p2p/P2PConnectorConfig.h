#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

struct P2PConnectorSchedulerConfig {
    std::vector<std::string> worker_grpc_addrs;
    std::vector<std::string> worker_addrs;
    int64_t                  p2p_transfer_not_done_resource_hold_ms       = 10 * 1000;
    int                      p2p_resource_store_timeout_check_interval_ms = 100;
    int64_t                  p2p_cancel_broadcast_timeout_ms              = 1000;

    static P2PConnectorSchedulerConfig create(const RuntimeConfig&    runtime_config,
                                              const CacheStoreConfig& cache_store_config,
                                              const PDSepConfig&      pd_sep_config) {
        P2PConnectorSchedulerConfig config;
        config.worker_grpc_addrs                      = runtime_config.worker_grpc_addrs;
        config.worker_addrs                           = runtime_config.worker_addrs;
        config.p2p_transfer_not_done_resource_hold_ms = cache_store_config.p2p_transfer_not_done_resource_hold_ms;
        config.p2p_resource_store_timeout_check_interval_ms =
            cache_store_config.p2p_resource_store_timeout_check_interval_ms;
        config.p2p_cancel_broadcast_timeout_ms = cache_store_config.p2p_cancel_broadcast_timeout_ms;
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
    // CP rank/size remain transport coordinates. FULL key ownership comes
    // from an external projection; first/stride index the global B-key array.
    // The worker owns this copied value for its lifetime. LayerCacheBuffer
    // copies selected keys before an async send starts.
    int  cp_rank          = 0;
    int  cp_size          = 1;
    bool kv_cache_sharded = false;
    std::optional<std::pair<size_t, size_t>> cache_key_coordinates;

    std::pair<size_t, size_t> resolvedCacheKeyCoordinates() const {
        if (!cache_key_coordinates.has_value()) {
            if (kv_cache_sharded && cp_size > 1) {
                throw std::invalid_argument("sharded P2P connector requires externally resolved cache key coordinates");
            }
            return {0, 1};
        }
        const auto [first, stride] = *cache_key_coordinates;
        if (stride == 0 || first >= stride) {
            throw std::invalid_argument("P2P connector cache key first/stride coordinates are invalid");
        }
        return *cache_key_coordinates;
    }

    static P2PConnectorWorkerConfig create(const CacheStoreConfig&  cache_store_config,
                                           const PDSepConfig&       pd_sep_config,
                                           const ParallelismConfig& parallelism_config,
                                           uint32_t                 layer_all_num,
                                           std::optional<std::pair<size_t, size_t>> cache_key_coordinates = std::nullopt) {
        P2PConnectorWorkerConfig config;
        config.transfer_backend_config.cache_store_rdma_mode         = cache_store_config.cache_store_rdma_mode;
        config.transfer_backend_config.rdma_transfer_wait_timeout_ms = cache_store_config.rdma_transfer_wait_timeout_ms;
        config.transfer_backend_config.messager_io_thread_count      = cache_store_config.messager_io_thread_count;
        config.transfer_backend_config.messager_worker_thread_count  = cache_store_config.messager_worker_thread_count;
        config.transfer_backend_config.rdma_max_block_pairs_per_connection =
            cache_store_config.rdma_max_block_pairs_per_connection;
        config.transfer_backend_config.cache_store_listen_port = pd_sep_config.cache_store_listen_port;
        config.transfer_backend_config.cache_store_tcp_anet_rpc_thread_num =
            cache_store_config.cache_store_tcp_anet_rpc_thread_num;
        config.transfer_backend_config.cache_store_tcp_anet_rpc_queue_num =
            cache_store_config.cache_store_tcp_anet_rpc_queue_num;
        config.p2p_layer_cache_buffer_store_timeout_ms = cache_store_config.p2p_layer_cache_buffer_store_timeout_ms;
        config.p2p_read_steal_before_deadline_ms       = cache_store_config.p2p_read_steal_before_deadline_ms;
        config.p2p_read_return_before_deadline_ms      = cache_store_config.p2p_read_return_before_deadline_ms;
        config.tp_size                                 = parallelism_config.tp_size;
        config.tp_rank                                 = parallelism_config.tp_rank;
        config.layer_all_num                           = layer_all_num;
        config.cache_key_coordinates                   = cache_key_coordinates;
        config.kv_cache_sharded                        = parallelism_config.prefill_cp_config.kv_cache_sharded;
        if (config.kv_cache_sharded && parallelism_config.tp_size > 1) {
            config.cp_size = static_cast<int>(parallelism_config.tp_size);
            config.cp_rank = static_cast<int>(parallelism_config.tp_rank);
        }
        (void)config.resolvedCacheKeyCoordinates();
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
                                     std::optional<std::pair<size_t, size_t>> cache_key_coordinates = std::nullopt) {
        P2PConnectorConfig config;
        config.role_type = pd_sep_config.role_type;
        config.tp_rank   = parallelism_config.tp_rank;
        config.scheduler_config =
            P2PConnectorSchedulerConfig::create(runtime_config, cache_store_config, pd_sep_config);
        config.worker_config =
            P2PConnectorWorkerConfig::create(cache_store_config, pd_sep_config, parallelism_config,
                                             layer_all_num, cache_key_coordinates);
        return config;
    }
};

}  // namespace rtp_llm
