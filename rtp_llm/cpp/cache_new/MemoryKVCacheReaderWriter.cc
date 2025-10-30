#include "rtp_llm/cpp/cache_new/MemoryKVCacheReaderWriter.h"

namespace rtp_llm {

bool MemoryKVCacheReaderWriter::init() {
    // TODO(LXQ): implement
    return true;
}

void MemoryKVCacheReaderWriter::asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    // TODO(LXQ): implement
}

void MemoryKVCacheReaderWriter::asyncReadByLayer(const BatchKVCacheResourcePtr& resource,
                                                 int                            layer_idx,
                                                 const CallBack&                callback) {
    // TODO(LXQ): implement
}

void MemoryKVCacheReaderWriter::asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    const auto& cache_keys = resource->cache_keys;
    for (const auto& cache_key : cache_keys) {
        if (match(cache_key)) {
            continue;
        }
        const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
        for (int layer_idx = 0; layer_idx < layer_block_indices.size(); layer_idx++) {
            const auto& block_indices_ptr = layer_block_indices[layer_idx];
            const auto& block_indices     = block_indices_ptr->block_indices;
            for (const auto& block_idx : block_indices) {
                const auto [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] =
                    allocator_->convertIndexToBuffer(layer_idx, block_idx);
                KVCacheConnector::Buffers connector_buffers;
                if (k_buffer) {
                    connector_buffers.push_back({k_buffer, cache_key, layer_idx, std::nullopt});
                }
                if (v_buffer) {
                    connector_buffers.push_back({v_buffer, cache_key, layer_idx, std::nullopt});
                }
                if (k_scale_buffer) {
                    connector_buffers.push_back({k_scale_buffer, cache_key, layer_idx, std::nullopt});
                }
                if (v_scale_buffer) {
                    connector_buffers.push_back({v_scale_buffer, cache_key, layer_idx, std::nullopt});
                }
                KVCacheConnector::Meta     meta;
                KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
                    block_indices_ptr.reset();
                    callback(success);
                };
                layer_to_connector_map_[layer_idx]->asyncPut(connector_buffers, meta, connector_callback);
            }
        }
    }
}

void MemoryKVCacheReaderWriter::asyncWriteByLayer(const BatchKVCacheResourcePtr& resource,
                                                  int                            layer_idx,
                                                  const CallBack&                callback) {
    const auto& cache_keys = resource->cache_keys[0];
    for (const auto& cache_key : cache_keys) {
        if (match(cache_key)) {
            continue;
        }
        const auto& layer_block_indices = resource->batch_cache_layer_layouts[0];
        // for (int layer_idx = 0; layer_idx < layer_block_indices.size(); layer_idx++) {
        const auto& block_indices_ptr = layer_block_indices[layer_idx];
        const auto& block_indices     = block_indices_ptr->block_indices;
        for (const auto& block_idx : block_indices) {
            const auto [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] =
                allocator_->convertIndexToBuffer(layer_idx, block_idx);
            KVCacheConnector::Buffers connector_buffers;
            if (k_buffer) {
                connector_buffers.push_back({k_buffer, cache_key, layer_idx, std::nullopt});
            }
            if (v_buffer) {
                connector_buffers.push_back({v_buffer, cache_key, layer_idx, std::nullopt});
            }
            if (k_scale_buffer) {
                connector_buffers.push_back({k_scale_buffer, cache_key, layer_idx, std::nullopt});
            }
            if (v_scale_buffer) {
                connector_buffers.push_back({v_scale_buffer, cache_key, layer_idx, std::nullopt});
            }
            KVCacheConnector::Meta     meta;
            KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
                block_indices_ptr.reset();
                callback(success);
            };
            layer_to_connector_map_[layer_idx]->asyncPut(connector_buffers, meta, connector_callback);
        }
        // }
    }
}

void MemoryKVCacheReaderWriter::connectorWrite(const std::shared_ptr<BlockIds>& block_indices_ptr,
                                               const CallBack&                  callback) {
    const auto& block_indices = block_indices_ptr->block_indices;
    for (const auto& block_idx : block_indices) {
        const auto [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] =
            allocator_->convertIndexToBuffer(layer_idx, block_idx);
        KVCacheConnector::Buffers connector_buffers;
        if (k_buffer) {
            connector_buffers.push_back({k_buffer, cache_key, layer_idx, std::nullopt});
        }
        if (v_buffer) {
            connector_buffers.push_back({v_buffer, cache_key, layer_idx, std::nullopt});
        }
        if (k_scale_buffer) {
            connector_buffers.push_back({k_scale_buffer, cache_key, layer_idx, std::nullopt});
        }
        if (v_scale_buffer) {
            connector_buffers.push_back({v_scale_buffer, cache_key, layer_idx, std::nullopt});
        }
        KVCacheConnector::Meta     meta;
        KVCacheConnector::CallBack connector_callback = [block_indices_ptr, callback](bool success) {
            block_indices_ptr.reset();
            callback(success);
        };
        layer_to_connector_map_[layer_idx]->asyncPut(connector_buffers, meta, connector_callback);
    }
}

bool MemoryKVCacheReaderWriter::match(int64_t key) const {
    for (const auto& c : connectors_) {
        if (!c->match(key)) {
            return false;
        }
    }
    return true;
}

// bool MemoryKVCacheReaderWriter::match(int64_t key) {
//     for (const auto& c : connectors_) {
//         if (c && c->match(key)) { return true; }
//     }
//     return false;
// }

// int32_t MemoryKVCacheReaderWriter::prefixMatch(const std::vector<int64_t>& keys) {
//     int32_t best = 0;
//     for (const auto& c : connectors_) {
//         if (!c) { continue; }
//         best = std::max(best, c->prefixMatch(keys));
//     }
//     return best;
// }

}  // namespace rtp_llm
