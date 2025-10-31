// #include <algorithm>
// #include <limits.h>

// #include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
// #include "rtp_llm/cpp/cache_new/BlockPool.h"
// #include "rtp_llm/cpp/cache_new/BlockCache.h"
// #include "rtp_llm/cpp/cache_new/KVCacheSpec.h"
// #include "rtp_llm/cpp/cache_new/LinearKVCacheGroup.h"
// #include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
// #include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
// #include "rtp_llm/cpp/utils/Logger.h"
// #include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
// #include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"

// namespace rtp_llm {

// HybridLayerKVCacheAllocator::HybridLayerKVCacheAllocator(const CacheConfig& config,
//                                                          rtp_llm::DeviceBase* device,
//                                                          AllocationType atype)
//     : KVCacheAllocator(config, device, atype), config_(config), device_(device), atype_(atype) {
//     RTP_LLM_LOG_INFO("Creating HybridLayerKVCacheAllocator with config: full_layer_num=%d, linear_layer_num=%d",
//                      config_.full_layer_num, config_.linear_layer_num);
// }

// bool HybridLayerKVCacheAllocator::init() {
//     if (!initKVCacheGroups()) {
//         RTP_LLM_LOG_ERROR("Failed to initialize KV cache groups");
//         return false;
//     }

//     RTP_LLM_LOG_INFO("HybridLayerKVCacheAllocator initialized successfully with %zu groups",
//                      kv_cache_groups_.size());
//     return true;
// }

// MallocResult HybridLayerKVCacheAllocator::malloc(const MallocInfo& malloc_info) {
//     auto batch_kv_cache_resource = malloc_info.batch_kv_cache_resource;

//     if(batch_kv_cache_resource->is_allocated_blocks || !batch_kv_cache_resource->enable_reuse_cache) {
//         return mallocSimple(malloc_info);
//     } else {
//         return mallocWithCache(malloc_info);
//     }

//     return MallocResult{true};
// }

// FreeResult HybridLayerKVCacheAllocator::free(const FreeInfo& free_info) {
//     auto batch_kv_cache_resource = free_info.batch_kv_cache_resource;

//     for(int i = 0; i < batch_kv_cache_resource->group_id_to_block_ids.size(); i++) {
//         auto group_id_to_block_ids = batch_kv_cache_resource->group_id_to_block_ids[i];
//         auto block_indices = group_id_to_block_ids->block_indices;
//         kv_cache_groups_[i]->free(block_indices);
//     }

//     return FreeResult{true};
// }

// InsertResult HybridLayerKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
//     // auto stream = insert_info.stream;
//     // if (!stream) {
//     //     RTP_LLM_LOG_ERROR("Invalid stream in insert info");
//     //     return {false};
//     // }

//     // const auto& kv_cache = stream->streamCacheResource()->kvCache();

//     // // 将已缓存在 block_cache 中的块插入到 block_cache 中
//     // for (size_t batch_id = 0; batch_id < kv_cache.cache_keys.size(); ++batch_id) {
//     //     const auto& cache_keys = kv_cache.cache_keys[batch_id];
//     //     if (batch_id < kv_cache.batch_block_id.size()) {
//     //         const auto& block_ids = kv_cache.batch_block_id[batch_id];

//     //         if (cache_keys.size() == block_ids.size() && !cache_keys.empty()) {
//     //             // 为每个 KVCacheGroup 插入缓存
//     //             for (auto& kv_cache_group : kv_cache_groups_) {
//     //                 std::vector<int64_t> group_cache_keys(cache_keys.begin(), cache_keys.end());
//     //                 std::vector<int> group_block_ids(block_ids.begin(), block_ids.end());
//     //                 kv_cache_group->insertIntoCache(group_cache_keys, group_block_ids);
//     //             }
//     //         }
//     //     }
//     // }

//     // RTP_LLM_LOG_DEBUG("Inserted cache for stream %ld", stream->streamId());
//     // return {true};

//     auto batch_kv_cache_resource = insert_info.batch_kv_cache_resource;
//     for(int i = 0; i < batch_kv_cache_resource->group_id_to_cached_block_ids.size(); i++) {
//         auto cached_block_ids = batch_kv_cache_resource->group_id_to_cached_block_ids[i]->block_indices;
//         auto block_ids = batch_kv_cache_resource->group_id_to_block_ids[i]->block_indices;

//         int cached_len = cached_block_ids.size();

//         vector<int> need_cached_block_ids(block_ids.begin() + cached_len, block_ids.end());
//         kv_cache_groups_[i]->insertIntoCache(need_cached_block_ids);
//     }

//     return InsertResult{true};
// }

// // d
// CacheLayerLayout HybridLayerKVCacheAllocator::layerCacheBase() const {
//     CacheLayerLayout layout;
//     layout.layer_to_groups.resize(config_.layer_num);

//     for (int i = 0; i < kv_cache_groups_.size(); ++i) {
//         std::unordered_map<int, torch::Tensor> layer_tensors = kv_cache_group_[i]->layerCacheBase();
//         for (auto& [layer_id, tensor] : layer_tensors) {
//             layout.layer_to_groups[layer_id] = i;
//             layout.layers_to_buffer_ptrs[layer_id] = tensor;
//         }
//     }

//     return layout;
// }

// // d
// BlockPoolPtr HybridLayerKVCacheAllocator::initSharedBlockPool(int layer_num) {
//     // 使用配置辅助类创建合适的布局配置
//     // 这里可以根据具体需求选择布局类型
//     BlockPoolConfig pool_config = BlockPoolConfigHelper::createLayerFirstConfig(
//         layer_num,
//         config_.block_num,
//         config_.block_size
//     );

//     BlockPoolPtr shared_block_pool = std::make_shared<BlockPool>(pool_config, device_, atype_);

//     if (!shared_block_pool->init()) {
//         RTP_LLM_LOG_ERROR("Failed to initialize shared block pool");
//         return nullptr;
//     }

//     RTP_LLM_LOG_INFO("Shared block pool initialized: layer_num=%d, block_num=%d, block_size=%d",
//                      pool_config.layer_num, pool_config.block_num, pool_config.block_size);
//     return shared_block_pool;
// }

// // d
// bool HybridLayerKVCacheAllocator::initKVCacheGroups() {
//     kv_cache_groups_.clear();

//     std::vector<int> layer_type_to_layer_num;
//     for (int i = 0; i < config_.layer_type_num; i++) {
//         int layer_num = config_.layer_ids[i].size();
//         layer_type_to_layer_num.push_back(layer_num);
//     }

//     int layer_num_per_group = *std::min_element(layer_type_to_layer_num.begin(), layer_type_to_layer_num.end());
//     if (layer_num_per_group <= 0) {
//         RTP_LLM_LOG_ERROR("Invalid minimum layer count: %d", layer_num_per_group);
//         return false;
//     }

//     shared_block_pool = initSharedBlockPool(layer_num_per_group);

//     RTP_LLM_LOG_INFO("Single group size (min layers): %d", layer_num_per_group);

//     std::vector<int> groups_per_type;

//     int group_id = 0;
//     for (int type_idx = 0; type_idx < config_.layer_type_num; type_idx++) {
//         int groups_needed = (layer_type_to_layer_num[type_idx] + layer_num_per_group - 1) / layer_num_per_group;
//         groups_per_type.push_back(groups_needed);
//         RTP_LLM_LOG_INFO("Attention type %d needs %d groups", type_idx, groups_needed);

//         const auto& type_param = config_.layer_type_params[type_idx];
//         const auto& type_layer_ids = config_.layer_ids[type_idx];

//         KVCacheGroupType cache_type;
//         switch (type_param.type) {
//             case rtp_llm::MultiHeadAttention:
//                 cache_type = rtp_llm::KVCacheGroupType::FULL;
//                 break;
//             case rtp_llm::MultiHeadLatentAttention:
//                 cache_type = rtp_llm::KVCacheGroupType::FULL;
//                 break;
//             case rtp_llm::LinearAttention:
//                 cache_type = rtp_llm::KVCacheGroupType::LINEAR;
//                 break;
//             default:
//                 RTP_LLM_LOG_ERROR("Unknown attention type: %d", static_cast<int>(type_param.type));
//                 return false;
//         }

//         for (int group_idx = 0; group_idx < groups_per_type[type_idx]; group_idx++) {
//             int start_layer = group_idx * layer_num_per_group;
//             int end_layer = std::min(start_layer + layer_num_per_group, static_cast<int>(type_layer_ids.size()));
//             vector<int> group_layer_ids(type_layer_ids.begin() + start_layer, type_layer_ids.begin() + end_layer);

//             for(auto& layer_id : group_layer_ids) {
//                 global_layer_to_group_id_[layer_id] = group_id;
//             }

//             KVCacheSpec group_spec;
//             group_spec.layer_ids_ = group_layer_ids;
//             group_spec.type_ = cache_type;

//             auto block_cache = std::make_shared<BlockCache>(1); // TODO(chanyin): edit after block_cache is
//             implemented

//             KVCacheGroupPtr kv_cache_group;
//             if (cache_type == rtp_llm::KVCacheGroupType::FULL) {
//                 kv_cache_group = std::make_shared<FullKVCacheGroup>(
//                     group_layer_ids, group_spec, block_cache, shared_block_pool);
//             } else {
//                 kv_cache_group = std::make_shared<LinearKVCacheGroup>(
//                     group_layer_ids, group_spec, block_cache, shared_block_pool);
//             }

//             kv_cache_groups_.push_back(kv_cache_group);
//             RTP_LLM_LOG_INFO("Created KVCacheGroup %d: type=%s, layers=%zu, layer_ids=[%s]",
//                             group_id,
//                             (cache_type == rtp_llm::KVCacheGroupType::FULL) ? "FULL" : "LINEAR",
//                             group_layer_ids.size(),
//                             [&]() {
//                                 std::string layer_str;
//                                 for (size_t i = 0; i < group_layer_ids.size(); ++i) {
//                                     if (i > 0) layer_str += ",";
//                                     layer_str += std::to_string(group_layer_ids[i]);
//                                 }
//                                 return layer_str;
//                             }().c_str());

//             group_id++;
//         }
//     }

//     RTP_LLM_LOG_INFO("Successfully created %zu KVCacheGroups", kv_cache_groups_.size());
//     return true;
// }

// MallocResult HybridLayerKVCacheAllocator::mallocWithCache(const MallocInfo& malloc_info) {
//     auto batch_kv_cache_resource = malloc_info.batch_kv_cache_resource;
//     auto complete_token_ids = malloc_info.complete_token_ids;

//     auto token_ids = complete_token_ids->commonCompleteTokenIdsVec(0);
//     auto cache_keys = batch_kv_cache_resource->cache_keys;

//     // 1. match in each kv cache group
//     int reuse_len = INT_MAX;
//     for (int group_id = 0; group_id < kv_cache_groups_.size(); group_id++) {
//         if(kv_cache_groups_[group_id]->type() == rtp_llm::KVCacheGroupType::FULL) {
//             auto match_result = kv_cache_groups_[group_id]->match(cache_keys);
//             if(match_result.reuse_length < reuse_len) {
//                 reuse_len = match_result.reuse_length;
//             }
//         }
//     }

//     std::vector<std::vector<int64_t>> linear_cached_keys;
//     for (int group_id = 0; group_id < kv_cache_groups_.size(); group_id++) {
//         if(kv_cache_groups_[group_id]->type() == rtp_llm::KVCacheGroupType::LINEAR) {
//             auto match_result = kv_cache_groups_[group_id]->match(cache_keys);
//             linear_cached_keys.push_back(match_result.cached_keys);
//         }
//     }

//     for (int i = reuse_len - 1; i >= 0; i--) {
//         bool found_in_all = true;
//         for (const auto& cached_keys : linear_cached_keys) {
//             if (cached_keys[i] == 0) {
//                 found_in_all = false;
//                 break;
//             }
//         }

//         if (found_in_all) {
//             reuse_len = i + 1;
//             break;
//         }
//     }

//     // 2. reference allop

//     return MallocResult{true};

//     // MallocResult malloc_result;
//     // malloc_result.success = false;

//     // auto stream = malloc_info.stream;
//     // const auto& cache_keys = stream->streamCacheResource()->kvCache().cache_keys;

//     // if (cache_keys.empty()) {
//     //     RTP_LLM_LOG_WARNING("No cache keys available for malloc with cache");
//     //     return mallocSimple(malloc_info);
//     // }

//     // // 获取第一个批次的缓存键（简化处理）
//     // const auto& batch_cache_keys = cache_keys[0];

//     // int full_reuse_len = INT_MAX;
//     // std::vector<MatchResult> match_results;

//     // // 为每个 KVCacheGroup 进行匹配
//     // for (auto& kv_cache_group : kv_cache_groups_) {
//     //     auto match_result = kv_cache_group->match(batch_cache_keys);
//     //     match_results.push_back(match_result);

//     //     // 如果是 FULL 类型的组，更新最小复用长度
//     //     if (kv_cache_group->type() == rtp_llm::KVCacheGroupType::FULL &&
//     //         static_cast<int>(match_result.reuse_length) < full_reuse_len) {
//     //         full_reuse_len = static_cast<int>(match_result.reuse_length);
//     //     }
//     // }

//     // // 计算实际的复用长度
//     // int reuse_len = 0;
//     // for (int i = full_reuse_len - 1; i >= 0; i--) {
//     //     bool found_in_all = true;
//     //     for (const auto& match_result : match_results) {
//     //         if (match_result.cached_keys.empty() ||
//     //             static_cast<size_t>(i) >= match_result.cached_keys[0].size() ||
//     //             match_result.cached_keys[0][i] != batch_cache_keys[i]) {
//     //             found_in_all = false;
//     //             break;
//     //         }
//     //     }
//     //     if (found_in_all) {
//     //         reuse_len = i + 1;
//     //         break;
//     //     }
//     // }

//     // // 更新 stream 的 BatchKVCacheResource
//     // auto& kv_cache = const_cast<BatchKVCacheResource&>(stream->streamCacheResource()->kvCache());

//     // // 为每个 KVCacheGroup 分配块
//     // for (size_t group_idx = 0; group_idx < kv_cache_groups_.size(); ++group_idx) {
//     //     auto& kv_cache_group = kv_cache_groups_[group_idx];

//     //     // 计算需要分配的块数量（总数减去复用的块数）
//     //     int needed_blocks = static_cast<int>(batch_cache_keys.size()) - reuse_len;
//     //     auto block_indices = kv_cache_group->malloc(needed_blocks);

//     //     // 如果有复用的块，需要将它们添加到结果中
//     //     if (reuse_len > 0 && group_idx < match_results.size() &&
//     //         !match_results[group_idx].block_indices.empty()) {
//     //         const auto& reused_blocks = match_results[group_idx].block_indices[0];
//     //         // 将复用的块插入到前面
//     //         block_indices.insert(block_indices.begin(), reused_blocks.begin(),
//     //                             reused_blocks.begin() + std::min(reuse_len,
//     static_cast<int>(reused_blocks.size())));
//     //     }

//     //     // 更新 BatchKVCacheResource
//     //     updateBatchKVCacheResource(kv_cache, group_idx, block_indices, match_results[group_idx]);
//     // }

//     // malloc_result.success = true;
//     // malloc_result.match_result.reuse_length = reuse_len;

//     // RTP_LLM_LOG_DEBUG("Malloc with cache completed: reuse_len=%d, groups=%zu",
//     //                   reuse_len, kv_cache_groups_.size());

//     // return malloc_result;
// }

// MallocResult HybridLayerKVCacheAllocator::mallocSimple(const MallocInfo& malloc_info) {
//     // MallocResult malloc_result;
//     // malloc_result.success = false;

//     // auto stream = malloc_info.stream;
//     // const auto& cache_keys = stream->streamCacheResource()->kvCache().cache_keys;

//     // if (cache_keys.empty()) {
//     //     RTP_LLM_LOG_ERROR("No cache keys available for simple malloc");
//     //     return malloc_result;
//     // }

//     // // 获取尚未分配块的缓存键
//     // const auto& batch_cache_keys = cache_keys[0];

//     // auto& kv_cache = const_cast<BatchKVCacheResource&>(stream->streamCacheResource()->kvCache());

//     // for (size_t group_idx = 0; group_idx < kv_cache_groups_.size(); ++group_idx) {
//     //     auto& kv_cache_group = kv_cache_groups_[group_idx];

//     //     // 简单分配，不使用缓存复用，分配所有需要的块
//     //     int needed_blocks = static_cast<int>(batch_cache_keys.size());
//     //     auto block_indices = kv_cache_group->malloc(needed_blocks);

//     //     // 更新 BatchKVCacheResource
//     //     MatchResult empty_match;
//     //     updateBatchKVCacheResource(kv_cache, group_idx, block_indices, empty_match);

//     //     // 如果是 LINEAR 类型，将之前的块插入到 block_cache 并释放
//     //     if (kv_cache_group->type() == KVCacheType::LINEAR) {
//     //         // 获取之前的块
//     //         if (!kv_cache.batch_block_id.empty() && !kv_cache.batch_block_id[0].empty()) {
//     //             std::vector<int> previous_blocks(kv_cache.batch_block_id[0].begin(),
//     //                                            kv_cache.batch_block_id[0].end());

//     //             // 插入到缓存并释放
//     //             kv_cache_group->insertIntoCache(batch_cache_keys, previous_blocks);
//     //             kv_cache_group->free(previous_blocks);
//     //         }
//     //     }
//     // }

//     // malloc_result.success = true;

//     // RTP_LLM_LOG_DEBUG("Simple malloc completed for %zu groups", kv_cache_groups_.size());

//     // return malloc_result;
// }

// void HybridLayerKVCacheAllocator::updateBatchKVCacheResource(BatchKVCacheResource& kv_cache,
//                                                            size_t group_idx,
//                                                            const std::vector<int>& block_indices,
//                                                            const MatchResult& match_result) {
//     // 确保 batch_cache_layer_layouts 有足够的空间
//     if (kv_cache.batch_cache_layer_layouts.size() <= group_idx) {
//         kv_cache.batch_cache_layer_layouts.resize(group_idx + 1);
//     }

//     // 确保每个批次都有对应的层布局
//     for (auto& batch_layouts : kv_cache.batch_cache_layer_layouts) {
//         if (batch_layouts.size() <= group_idx) {
//             batch_layouts.resize(group_idx + 1);
//         }
//         if (!batch_layouts[group_idx]) {
//             batch_layouts[group_idx] = std::make_shared<BlockIds>();
//         }
//     }

//     // 更新块索引
//     if (!kv_cache.batch_cache_layer_layouts.empty() &&
//         kv_cache.batch_cache_layer_layouts[0].size() > group_idx) {
//         kv_cache.batch_cache_layer_layouts[0][group_idx]->block_indices = block_indices;
//     }

//     // 如果有匹配结果，更新缓存的布局
//     if (match_result.reuse_length > 0 && !match_result.block_indices.empty()) {
//         // 确保 batch_cache_layer_cached_layouts 有足够的空间
//         if (kv_cache.batch_cache_layer_cached_layouts.size() <= group_idx) {
//             kv_cache.batch_cache_layer_cached_layouts.resize(group_idx + 1);
//         }

//         for (auto& batch_cached_layouts : kv_cache.batch_cache_layer_cached_layouts) {
//             if (batch_cached_layouts.size() <= group_idx) {
//                 batch_cached_layouts.resize(group_idx + 1);
//             }
//             if (!batch_cached_layouts[group_idx]) {
//                 batch_cached_layouts[group_idx] = std::make_shared<BlockIds>();
//             }
//         }

//         // 更新缓存的块索引
//         if (!kv_cache.batch_cache_layer_cached_layouts.empty() &&
//             kv_cache.batch_cache_layer_cached_layouts[0].size() > group_idx) {
//             kv_cache.batch_cache_layer_cached_layouts[0][group_idx]->block_indices = match_result.block_indices[0];
//         }
//     }
// }

// }  // namespace rtp_llm
