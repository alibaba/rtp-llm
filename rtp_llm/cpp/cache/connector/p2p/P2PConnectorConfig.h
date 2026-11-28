#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
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

// Pool lookups use module-local layer IDs. The PD protocol concatenates the
// physical modules on each stage and numbers wire keys across the whole model.
struct P2PModuleLayers {
    std::map<std::string, uint32_t> stage_offsets;

    uint32_t stageOffset(const std::string& tag) const {
        const auto it = stage_offsets.find(tag);
        return it == stage_offsets.end() ? 0 : it->second;
    }
};

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
    // Tag -> owning cache module, derived from CacheConfig by the multi-module
    // production factory. Empty means the legacy single-module (module 0) view.
    std::map<std::string, size_t>        module_by_tag;
    std::vector<std::string>             block_table_tags;
    P2PModuleLayers                      module_layers;
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
    P2PModuleLayers module_layers;

    uint32_t wireLayerId(const std::string& tag, uint32_t module_local_id) const {
        return global_layer_begin + module_layers.stageOffset(tag) + module_local_id;
    }

    std::shared_ptr<const CacheTopology> topology;
    size_t                               cache_key_tokens_per_block = 1;
    // Prefill page-RR shard geometry. cp_size==1 disables RR remap (legacy path);
    // cp_size>1 makes LayerCacheBufferUtil register cache_keys[cp_rank + i*cp_size]
    // for the i-th rank-local owned block (Stage 4 fix).
    int  cp_rank          = 0;
    int  cp_size          = 1;
    bool kv_cache_sharded = false;

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
        config.kv_cache_sharded                        = parallelism_config.prefill_cp_config.kv_cache_sharded;
        if (config.kv_cache_sharded && parallelism_config.tp_size > 1) {
            config.cp_size = static_cast<int>(parallelism_config.tp_size);
            config.cp_rank = static_cast<int>(parallelism_config.tp_rank);
        }
        return config;
    }
};

struct P2PConnectorConfig {
    RoleType role_type = RoleType::PDFUSION;
    int      tp_rank   = 0;

    P2PConnectorSchedulerConfig scheduler_config;
    P2PConnectorWorkerConfig    worker_config;

    // Build the P2P transfer view for a multi-module cache configuration.
    //
    // Every module keeps its own module-local topology and pools; the
    // P2P layer needs one executable view that can validate and address any
    // module's buffers. This view only unions the groups (tags are globally
    // unique after the MTP rename) and, per module-local layer id, the union of
    // tags that own that layer. The owning module for a tag is still resolved
    // through LayerBlockConverter::moduleOfTag at buffer construction, so this
    // is a transfer-only address space, never a shared cache topology.
    static std::shared_ptr<const CacheTopology> buildTransferTopology(const CacheConfig& cache_config) {
        std::vector<GroupBase>   groups;
        std::map<int, LayerBase> merged_layers;
        groups.reserve(static_cast<size_t>(cache_config.allGroupNums()));
        for (size_t module_id = 0; module_id < cache_config.moduleCount(); ++module_id) {
            const auto& module_topology = cache_config.topology(module_id);
            for (const auto& group : module_topology.groups()) {
                if (!module_topology.layerIdsForGroup(group.tag).empty()) {
                    groups.push_back(group);
                }
            }
            for (const auto& layer : module_topology.layers()) {
                auto& merged    = merged_layers[layer.layer_id];
                merged.layer_id = layer.layer_id;
                for (const auto& tag : layer.group_tags) {
                    if (std::find(merged.group_tags.begin(), merged.group_tags.end(), tag)
                        == merged.group_tags.end()) {
                        merged.group_tags.push_back(tag);
                    }
                }
            }
        }
        std::vector<LayerBase> layers;
        layers.reserve(merged_layers.size());
        for (auto& [layer_id, layer] : merged_layers) {
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(layer_id) == layers.size(),
                "P2P transfer topology requires contiguous module-local layer ids; got layer_id=%d at index=%zu",
                layer_id,
                layers.size());
            layers.push_back(std::move(layer));
        }
        return CacheTopology::create(std::move(groups), std::move(layers));
    }

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

    // Multi-module production factory: install the transfer view derived from
    // every module so draft (mtp_*) tags are valid for route
    // validation, payload lookup, and expected-buffer accounting.
    static P2PConnectorConfig create(const RuntimeConfig&     runtime_config,
                                     const CacheStoreConfig&  cache_store_config,
                                     const ParallelismConfig& parallelism_config,
                                     const PDSepConfig&       pd_sep_config,
                                     uint32_t                 layer_all_num,
                                     size_t                   llm_kv_block_size_bytes,
                                     const CacheConfig&       cache_config) {
        P2PConnectorConfig config = create(runtime_config,
                                           cache_store_config,
                                           parallelism_config,
                                           pd_sep_config,
                                           layer_all_num,
                                           llm_kv_block_size_bytes);
        auto topology                    = buildTransferTopology(cache_config);
        config.scheduler_config.topology = topology;
        config.worker_config.topology    = std::move(topology);
        config.scheduler_config.block_table_tags = cache_config.allGroupTags();
        auto& linear_specs = config.worker_config.linear_specs_by_layer;
        linear_specs.resize(config.worker_config.topology->layers().size());
        for (const auto& layer : config.worker_config.topology->layers()) {
            for (const auto& tag : layer.group_tags) {
                const auto& spec = config.worker_config.topology->group(tag).spec;
                if (spec->tp_transfer.mode == TpCacheTransferMode::LINEAR_QKV) {
                    linear_specs[layer.layer_id].emplace(tag, spec);
                }
            }
        }
        uint32_t stage_offset = 0;
        for (size_t module_id = 0; module_id < cache_config.moduleCount(); ++module_id) {
            const auto& module_topology = cache_config.topology(module_id);
            for (const auto& tag : module_topology.groupTags()) {
                RTP_LLM_CHECK_WITH_INFO(config.worker_config.module_layers.stage_offsets.emplace(tag, stage_offset).second,
                                        "P2P transfer topology duplicate cache tag=%s",
                                        tag.c_str());
                if (!module_topology.layerIdsForGroup(tag).empty()) {
                    config.scheduler_config.module_by_tag.emplace(tag, module_id);
                }
            }
            stage_offset += module_topology.layers().size();
        }
        config.scheduler_config.module_layers = config.worker_config.module_layers;
        return config;
    }
};

}  // namespace rtp_llm
