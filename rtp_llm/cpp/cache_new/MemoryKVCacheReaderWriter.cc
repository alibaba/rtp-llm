#include "rtp_llm/cpp/cache_new/MemoryKVCacheReaderWriter.h"

namespace rtp_llm {

bool MemoryKVCacheReaderWriter::init() {
    // TODO(LXQ): implement
    return true;
}

void MemoryKVCacheReaderWriter::asyncRead(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
        if (callback) {
            callback(true);
        }
        return;
    }

    const auto& cache_keys    = resource->cache_keys[0];
    const auto  gpu_reuse_len = resource->reuse_cache_key_num;
    for (size_t i = gpu_reuse_len; i < cache_keys.size(); i++) {
        const auto& cache_key    = cache_keys.at(i);
        const auto& buffers      = cacheKeyWiseLayout(cache_key, resource);
        auto        new_callback = [callback, resource](bool success) {
            resource.reset();
            if (callback) {
                callback(success);
            }
        };
        readBuffers(buffers, new_callback);
    }
}

void MemoryKVCacheReaderWriter::readBuffers(const KVCacheConnector::Buffers& buffers, const CallBack& callback) const {
    std::map<int, Buffers> group_to_buffers;
    for (const auto& buffer : buffers) {
        group_to_buffers.at(buffer.key1).push_back(buffer);
    }
    for (const auto& [group_id, buffers] : group_to_buffers) {
        KVCacheConnector::Meta meta;
        if (group_type_map_.at(group_id) == KVCacheGroupType::Full) {
            group_to_connector_.at(group_id)->asyncPrefixGet(buffers, meta, callback);
        } else if (group_type_map_.at(group_id) == KVCacheGroupType::Linear) {
            group_to_connector_.at(group_id)->asyncGet(buffers, meta, callback);
        } else {
            RTP_LLM_LOG_ERROR("invalid group type: {}", static_cast<int>(group_type_map_.at(group_id)));
            callback(false);
        }
    }
}

void MemoryKVCacheReaderWriter::asyncWrite(const BatchKVCacheResourcePtr& resource, const CallBack& callback) {
    if (!resource || resource->cache_keys.empty() || resource->batch_cache_layer_layouts.empty()) {
        if (callback) {
            callback(true);
        }
        return;
    }

    const auto& cache_keys = resource->cache_keys[0];
    const auto& match_len  = match(cache_keys);
    for (int i = match_len; i < cache_keys.size(); i++) {
        const auto& cache_key    = cache_keys.at(i);
        const auto& buffers      = cacheKeyWiseLayout(cache_key, resource);
        auto        new_callback = [callback, resource](bool success) {
            resource.reset();
            if (callback) {
                callback(success);
            }
        };
        writeBuffers(buffers, new_callback);
    }
}

void MemoryKVCacheReaderWriter::writeBuffers(const KVCacheConnector::Buffers& buffers, const CallBack& callback) const {
    std::map<int, Buffers> group_to_buffers;
    for (const auto& buffer : buffers) {
        group_to_buffers.at(buffer.key1).push_back(buffer);
    }
    for (const auto& [group_id, buffers] : group_to_buffers) {
        KVCacheConnector::Meta meta;
        if (group_type_map_.at(group_id) == KVCacheGroupType::Full) {
            group_to_connector_.at(group_id)->asyncPrefixPut(buffers, meta, callback);
        } else if (group_type_map_.at(group_id) == KVCacheGroupType::Linear) {
            group_to_connector_.at(group_id)->asyncPut(buffers, meta, callback);
        } else {
            RTP_LLM_LOG_ERROR("invalid group type: {}", static_cast<int>(group_type_map_.at(group_id)));
            callback(false);
        }
    }
}

KVCacheConnector::Buffers MemoryKVCacheReaderWriter::cacheKeyWiseLayout(int64_t                        cache_key,
                                                                        const BatchKVCacheResourcePtr& resource) const {
    KVCacheConnector::Buffers connector_buffers;
    const auto&               cache_keys          = resource->cache_keys.at(0);
    const auto&               layer_block_indices = resource->batch_cache_layer_layouts.at(0);
    // 第一维是cache_key
    for (int cache_key_idx = 0; cache_key_idx < cache_keys.size(); cache_key_idx++) {
        if (cache_keys.at(cache_key_idx) != cache_key) {
            continue;
        }
        // const auto& cache_key = cache_keys.at(cache_key_idx);
        // 第二维是layer
        for (int layer_idx = 0; layer_idx < layer_block_indices.size(); layer_idx++) {
            const auto& block_idx = layer_block_indices.at(layer_idx)->block_indices.at(cache_key_idx);
            const auto  group_idx = layer_to_group_.at(layer_idx);
            // 第三维是buffer
            const auto& [k_buffer, v_buffer, k_scale_buffer, v_scale_buffer] =
                allocator_->convertIndexToBuffer(layer_idx, block_idx);
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

size_t MemoryKVCacheReaderWriter::match(const std::vector<int64_t>& keys) const {
    int         match_len          = prefixMatch(keys);
    const auto& keys_to_hash_match = std::vector<int64_t>(keys.begin(), keys.begin() + match_len);
    const auto& hash_match_result  = hashMatch(keys_to_hash_match);
    for (size_t i = match_len - 1; i >= 0; i--) {
        if (hash_match_result[i]) {
            match_len = i + 1;
            break;
        }
    }
    return match_len;
}

size_t MemoryKVCacheReaderWriter::prefixMatch(const std::vector<int64_t>& keys) const {
    size_t match_len = keys.size();
    for (const auto& [group_id, connector] : group_to_connector_map_) {
        if (group_type_map_.at(group_id) == KVCacheGroupType::Full) {
            const auto& prefix_match_len = connector->prefixMatch(keys);
            if (prefix_match_len == 0) {
                return 0;
            }
            if (prefix_match_len < match_len) {
                match_len = prefix_match_len;
            }
        }
    }
    return match_len;
}

std::vector<bool> MemoryKVCacheReaderWriter::hashMatch(const std::vector<int64_t>& keys) const {
    std::vector<bool> match_result(keys.size(), true);
    for (const auto& [group_id, connector] : group_to_connector_map_) {
        if (group_type_map_.at(group_id) == KVCacheGroupType::Linear) {
            const auto& hash_match_result = connector->match(keys);
            for (size_t i = 0; i < match_result.size(); i++) {
                match_result[i] &= hash_match_result[i];
            }
        }
    }
    return match_result;
}

// // [layer_id, cache_key, buffers]
// std::vector<std::pair<int32_t, Buffers>> MemoryKVCacheReaderWriter::layerWiseLayout(const BatchKVCacheResourcePtr&
// resource, int layer_idx) const {
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
