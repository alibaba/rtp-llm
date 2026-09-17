#pragma once

#include <set>
#include <sstream>
#include <stdexcept>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV41CacheState.h"
#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

inline CacheKeysType dsv41PrefillCacheKeysForLoad(const GenerateRequestPB& request,
                                                 int64_t                  request_id,
                                                 int64_t                  input_length,
                                                 size_t                   tokens_per_block) {
    if (request.stage() != RemoteStage::LOAD || request.request_id() != request_id
        || input_length <= 0 || input_length > 1048576 || tokens_per_block == 0
        || request.prefill_cache_keys_size()
               != 1 + (static_cast<size_t>(input_length) - 1) / tokens_per_block)
        throw std::invalid_argument("V4.1 PD LOAD requires the producer keys for every prompt block");
    return {request.prefill_cache_keys().begin(), request.prefill_cache_keys().end()};
}

inline DSV41CacheIdentity dsv41CacheIdentity(const CacheConfig& config) {
    const auto* swa = dynamic_cast<const DSV41KVCacheSpec*>(config.cache_specs.at(5).get());
    if (config.dsv41_cache_layout_version != 1 || !swa || swa->region != KVCacheRegionName::SWA_KV
        || (config.dsv41_replay_mode != "full" && config.dsv41_replay_mode != "bounded_checkpoint_v1"))
        throw std::invalid_argument("V4.1 transfer requires an explicit supported cache layout and replay mode");
    DSV41CacheIdentity identity{config.dsv41_model_revision,
                                config.dsv41LayoutFingerprint(),
                                config.dsv41_replay_mode == "full" ? DSV41ReplayMode::FULL :
                                                                     DSV41ReplayMode::BOUNDED_CHECKPOINT_V1,
                                config.dsv41_tail_policy_version,
                                128,
                                swa->entries_per_block};
    return identity;
}

// Call before enqueue: the scheduler may finish P and remove its partial
// cache key while the RPC thread is still preparing LOAD. Keep the existing
// canonical token/image hashing and the producer's local layout identity.
inline CacheKeysType dsv41PrefillPromptCacheKeys(const CompleteTokenIdsPtr& tokens, const CacheConfig& config) {
    if (!tokens || tokens->seqLength() <= 0 || tokens->seqLength() > 1048576 || config.seq_size_per_block == 0)
        throw std::invalid_argument("V4.1 PD requires a complete canonical prompt before enqueue");
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->cacheResource().setDsv41CacheKeySeed(dsv41CacheIdentity(config).cacheKeySeed());
    initCacheKeys(resource, tokens, config.seq_size_per_block);
    return resource->cacheKeys();
}

inline V41TransferIdentityPB dsv41TransferIdentity(const CacheConfig& config, int cp_size) {
    const auto local = dsv41CacheIdentity(config);
    if (config.layer_num != 40 || (config.layer_all_num != 40 && config.layer_all_num != 43)
        || cp_size < 1)
        throw std::invalid_argument("V4.1 transfer requires target40 and either zero or three draft layers");
    std::ostringstream layout;
    std::ostringstream sources;
    layout << "dsv41-pd-layout-v1:block=" << config.seq_size_per_block << ":kernel=" << config.kernel_seq_size_per_block
           << ":layers=" << config.layer_all_num << ';';
    sources << "dsv41-csa2-ced-sources-v1;";
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        const auto* spec = dynamic_cast<const DSV41KVCacheSpec*>(config.cache_specs[gid].get());
        if (!spec || spec->cp_size != static_cast<uint32_t>(cp_size))
            throw std::invalid_argument("V4.1 transfer cache spec or CP geometry mismatch");
        // P fixed pools contain byte slices; the wire identity names the full D page.
        layout << gid << ':' << static_cast<int>(spec->region) << ':' << static_cast<int>(spec->encoding) << ':'
               << spec->ratio << ':' << spec->entry_bytes << ':' << spec->entries_per_block << ':'
               << spec->quant_group_size << ':' << spec->seq_size_per_block << ':' << spec->cp_size << ':'
               << spec->full_block_size_bytes() << ';';
        sources << gid << ':';
        for (int layer : config.global_layer_ids.at(gid))
            sources << layer << ',';
        sources << ';';
    }
    for (size_t layer = 0; layer < config.layer_region_to_owner.size(); ++layer) {
        sources << layer << ':';
        for (int owner : config.layer_region_to_owner[layer])
            sources << owner << ',';
        sources << ';';
    }
    sources << "topk:";
    for (int owner : config.dsv41_topk_owner)
        sources << owner << ',';
    V41TransferIdentityPB wire;
    wire.set_schema_version(1);
    wire.set_model_revision(local.model_revision);
    wire.set_layout_fingerprint(layout.str());
    wire.set_source_fingerprint(sources.str());
    wire.set_replay_mode(config.dsv41_replay_mode);
    wire.set_tail_policy_version(local.tail_policy_version);
    wire.set_replay_window(local.replay_window);
    wire.set_physical_swa_entries(local.physical_swa_entries);
    wire.set_draft_layers(config.layer_all_num - config.layer_num);
    wire.set_prefill_cp_size(cp_size);
    return wire;
}

inline void validateDSV41TransferIdentity(const CacheConfig& config, const V41TransferIdentityPB& wire, int cp_size) {
    const auto expected = dsv41TransferIdentity(config, cp_size);
    if (wire.schema_version() != expected.schema_version() || wire.model_revision() != expected.model_revision()
        || wire.layout_fingerprint() != expected.layout_fingerprint()
        || wire.source_fingerprint() != expected.source_fingerprint() || wire.replay_mode() != expected.replay_mode()
        || wire.tail_policy_version() != expected.tail_policy_version()
        || wire.replay_window() != expected.replay_window()
        || wire.physical_swa_entries() != expected.physical_swa_entries()
        || wire.draft_layers() != expected.draft_layers() || wire.prefill_cp_size() != expected.prefill_cp_size())
        throw std::invalid_argument("V4.1 PD model/layout/source/mode/tail/CP identity mismatch");
}

inline void validateDSV41Peers(const std::vector<std::string>& peers, int cp_size) {
    if (cp_size <= 0 || peers.size() != static_cast<size_t>(cp_size)
        || std::set<std::string>(peers.begin(), peers.end()).size() != peers.size())
        throw std::invalid_argument("V4.1 PD requires every distinct prefill CP shard exactly once");
}

inline size_t dsv41FixedDestinationSliceBytes(const CacheConfig& config, size_t group, size_t cp_size) {
    const auto* spec = dynamic_cast<const DSV41KVCacheSpec*>(config.cache_specs.at(group).get());
    if (config.dsv41_cache_layout_version != 1 || !spec || spec->prefill_byte_slice || cp_size == 0
        || spec->cp_size != cp_size
        || (spec->region != KVCacheRegionName::SWA_KV && spec->region != KVCacheRegionName::DSV41_PAIR_STATE)
        || spec->full_block_size_bytes() % cp_size != 0)
        throw std::invalid_argument("V4.1 PD fixed state requires full destination pages and matching CP slices");
    return spec->full_block_size_bytes() / cp_size;
}

}  // namespace rtp_llm
