#include "rtp_llm/cpp/cache_new/MemoryKVCacheReaderWriter.h"

namespace rtp_llm {

bool MemoryKVCacheReaderWriter::init() {
    // TODO(LXQ): implement
    return true;
}

void MemoryKVCacheReaderWriter::asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
        if (callback) { callback(true); }
        return;
    }
    const auto& cache_keys = resource->cache_keys[0];
    const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
    for (const auto& cache_key : cache_keys) {
        for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_block_indices.size()); ++layer_idx) {
            const auto& block_indices_ptr = layer_block_indices[layer_idx];
            if (!block_indices_ptr) { continue; }
            asyncReadLayer(layer_idx, cache_key, block_indices_ptr, callback);
        }
    }
}

// void MemoryKVCacheReaderWriter::asyncReadByLayer(const BatchKVCacheResourcePtr& resource,
//                                                  int                            layer_idx,
//                                                  const CallBack&                callback) {
//     if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
//         if (callback) { callback(true); }
//         return;
//     }
//     const auto& cache_keys = resource->cache_keys[0];
//     const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
//     if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_block_indices.size())) {
//         if (callback) { callback(false); }
//         return;
//     }
//     const auto& block_indices_ptr = layer_block_indices[layer_idx];
//     if (!block_indices_ptr) {
//         if (callback) { callback(true); }
//         return;
//     }
//     for (const auto& cache_key : cache_keys) {
//         asyncReadLayer(layer_idx, cache_key, block_indices_ptr, callback);
//     }
// }

void MemoryKVCacheReaderWriter::asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
        if (callback) { callback(true); }
        return;
    }

    const auto& cache_keys = resource->cache_keys[0];
    for (const auto& cache_key : cache_keys) {
        if (match(cache_key)) { continue; }
        const auto & buffers = cacheKeyWiseLayout(cache_key, resource);
        
    }
}

void MemoryKVCacheReaderWriter::writeBuffers(const KVCacheConnector::Buffers& buffers, const CallBack& callback) const {
    std::map<int, Buffers> group_to_buffers;
    for (const auto& buffer : buffers) {
        group_to_buffers.at(buffer.key1).push_back(buffer);
    }
    for (const auto& [group_id, buffers] : group_to_buffers) {
        KVCacheConnector::Meta meta;
        if (group_type_map_.at(group_id) == KVCacheGroupType::FULL) {
            group_to_connector_.at(group_id)->asyncPrefixPut(buffers, meta, callback);
        } else if (group_type_map_.at(group_id) == KVCacheGroupType::LINEAR) {
            group_to_connector_.at(group_id)->asyncPut(buffers, meta, callback);
        } else {
            RTP_LLM_LOG_ERROR("invalid group type: {}", static_cast<int>(group_type_map_.at(group_id)));
            callback(false);
        }
    }
}

// void MemoryKVCacheReaderWriter::asyncWriteByLayer(const BatchKVCacheResourcePtr& resource,
//                                                   int                            layer_idx,
//                                                   const CallBack&                callback) {
//     if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
//         if (callback) { callback(true); }
//         return;
//     }
//     const auto& cache_keys = resource->cache_keys[0];
//     const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
//     if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_block_indices.size())) {
//         if (callback) { callback(false); }
//         return;
//     }
//     const auto& block_indices_ptr = layer_block_indices[layer_idx];
//     if (!block_indices_ptr) {
//         if (callback) { callback(true); }
//         return;
//     }
//     for (const auto& cache_key : cache_keys) {
//         if (match(cache_key)) { continue; }
//         asyncWriteLayer(layer_idx, cache_key, block_indices_ptr, callback);
//     }
// }

// void MemoryKVCacheReaderWriter::asyncWriteLayer(int32_t layer_idx, int64_t cache_key, const std::shared_ptr<BlockIds>& block_indices_ptr, const CallBack& callback) {
//     if (!allocator_) { if (callback) { callback(false); } return; }
//     const auto& block_indices = block_indices_ptr->block_indices;
//     KVCacheConnector::Buffers connector_buffers;
//     connector_buffers.reserve(block_indices.size() * 4);
//     for (const auto& block_idx : block_indices) {
//         auto bi = allocator_->convertIndexToBuffer(layer_idx, block_idx);
//         if (bi.k_addr)        connector_buffers.push_back({cache_key, bi.k_addr, BufferType::K, std::nullopt});
//         if (bi.v_addr)        connector_buffers.push_back({cache_key, bi.v_addr, BufferType::V, std::nullopt});
//         if (bi.k_scale_addr)  connector_buffers.push_back({cache_key, bi.k_scale_addr, BufferType::K_SCALE, std::nullopt});
//         if (bi.v_scale_addr)  connector_buffers.push_back({cache_key, bi.v_scale_addr, BufferType::V_SCALE, std::nullopt});
//     }
//     KVCacheConnector::Meta     meta;
//     KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
//         (void)block_indices_ptr;  // keep shared ownership until async completes
//         if (callback) { callback(success); }
//     };
//     auto it = layer_to_connector_map_.find(layer_idx);
//     if (it != layer_to_connector_map_.end() && it->second) {
//         it->second->asyncPut(connector_buffers, meta, connector_callback);
//     } else {
//         if (callback) { callback(false); }
//     }
// }

// void MemoryKVCacheReaderWriter::asyncReadLayer(int64_t cache_key, int32_t layer_idx, const std::shared_ptr<BlockIds>& block_indices_ptr, const CallBack& callback) {
//     if (!allocator_) { if (callback) { callback(false); } return; }

//     const auto& block_indices = block_indices_ptr->block_indices;
//     KVCacheConnector::Buffers connector_buffers;
//     connector_buffers.reserve(block_indices.size() * 4);
//     for (const auto& block_idx : block_indices) {
//         auto bi = allocator_->convertIndexToBuffer(layer_idx, block_idx);
//         if (bi.k_addr)        connector_buffers.push_back({cache_key, bi.k_addr, BufferType::K, std::nullopt});
//         if (bi.v_addr)        connector_buffers.push_back({cache_key, bi.v_addr, BufferType::V, std::nullopt});
//         if (bi.k_scale_addr)  connector_buffers.push_back({cache_key, bi.k_scale_addr, BufferType::K_SCALE, std::nullopt});
//         if (bi.v_scale_addr)  connector_buffers.push_back({cache_key, bi.v_scale_addr, BufferType::V_SCALE, std::nullopt});
//     }
//     KVCacheConnector::Meta     meta;
//     KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
//         (void)block_indices_ptr;
//         if (callback) { callback(success); }
//     };
//     auto it = layer_to_connector_map_.find(layer_idx);
//     if (it != layer_to_connector_map_.end() && it->second) {
//         it->second->asyncGet(connector_buffers, meta, connector_callback);
//     } else {
//         if (callback) { callback(false); }
//     }
// }

bool MemoryKVCacheReaderWriter::match(int64_t key) const {
    for (const auto& [group_id, connector] : group_to_connector_map_) {
        if (!connector->match(key)) {
            return false;
        }
    }
    return true;
}

KVCacheConnector::Buffers MemoryKVCacheReaderWriter::cacheKeyWiseLayout(int64_t cache_key, const BatchKVCacheResourcePtr& resource) const {
    KVCacheConnector::Buffers connector_buffers;
    const auto& cache_keys = resource->cache_keys.at(0);
    const auto& layer_block_indices = resource->batch_cache_layer_layouts.at(0);
    // 第一维是cache_key
    for (int cache_key_idx = 0; cache_key_idx < cache_keys.size(); cache_key_idx++) {
        if (cache_keys.at(cache_key_idx) != cache_key) { 
            continue; 
        }
        // const auto& cache_key = cache_keys.at(cache_key_idx);
        // 第二维是layer
        for (int layer_idx = 0; layer_idx < layer_block_indices.size(); layer_idx++) {
            const auto& block_idx = layer_block_indices.at(layer_idx)->block_indices.at(cache_key_idx);
            const auto group_idx = layer_to_group_.at(layer_idx);
            // 第三维是buffer
            const auto& [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] = allocator_->convertIndexToBuffer(layer_idx, block_idx);
            if (k_buffer) {
                connector_buffers.push_back({group_idx, cache_key, k_buffer, std::nullopt});
            }
            if (v_buffer) {
                connector_buffers.push_back({group_idx, cache_key, v_buffer, std::nullopt});
            }
            if (k_scale_buffer) {
                connector_buffers.push_back({group_idx, cache_key, k_scale_buffer, std::nullopt});
            }
            if (v_scale_buffer) {
                connector_buffers.push_back({group_idx, cache_key, v_scale_buffer, std::nullopt});
            }
        }
    }
    return connector_buffers;
}

// // [layer_id, cache_key, buffers]
// std::vector<std::pair<int32_t, Buffers>> MemoryKVCacheReaderWriter::layerWiseLayout(const BatchKVCacheResourcePtr& resource, int layer_idx) const {
//     std::vector<std::pair<int, Buffers>> layer_buffers;
//     const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
//     for (int layer_idx = 0; layer_idx < layer_block_indices.size(); layer_idx++) {
//         KVCacheConnector::Buffers connector_buffers;
//         const auto& block_indices     = layer_block_indices[layer_idx]->block_indices;
//         for (const auto& block_idx : block_indices) {
//             const auto [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] =
//                 allocator_->convertIndexToBuffer(layer_idx, block_idx);
//             if (k_buffer) {
//                 connector_buffers.push_back({k_buffer, cache_key, layer_idx, std::nullopt});
//             }
//             if (v_buffer) {
//                 connector_buffers.push_back({v_buffer, cache_key, layer_idx, std::nullopt});
//             }
//             if (k_scale_buffer) {
//                 connector_buffers.push_back({k_scale_buffer, cache_key, layer_idx, std::nullopt});
//             }
//             if (v_scale_buffer) {
//                 connector_buffers.push_back({v_scale_buffer, cache_key, layer_idx, std::nullopt});
//             }
//         }
//         layer_buffers.push_back({layer_idx, connector_buffers});
//         // KVCacheConnector::Meta     meta;
//         // KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
//         //     block_indices_ptr.reset();
//         //     callback(success);
//         // };
//         // layer_to_connector_map_[layer_idx]->asyncPut(connector_buffers, meta, connector_callback);
//     }
// }

}  // namespace rtp_llm
