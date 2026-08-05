#include "rtp_llm/cpp/models/ModelTypes.h"
#include <algorithm>
#include <cstring>
#include <map>
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace rtp_llm {

namespace {

uint64_t cacheGroupSchemaHash(const std::vector<int32_t>& wire) {
    constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
    constexpr uint64_t kFnvPrime  = 1099511628211ULL;
    uint64_t           hash       = kFnvOffset;
    for (const auto word : wire) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&word);
        for (size_t i = 0; i < sizeof(word); ++i) {
            hash ^= bytes[i];
            hash *= kFnvPrime;
        }
    }
    return hash;
}

bool sameCacheGroupSchema(const std::vector<CacheGroupHint>& lhs, const std::vector<CacheGroupHint>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].tag != rhs[i].tag || lhs[i].type != rhs[i].type) {
            return false;
        }
    }
    return true;
}

CacheGroupSchemaCache& processCacheGroupSchemaCache() {
    static CacheGroupSchemaCache cache;
    return cache;
}

}  // namespace

bool CacheGroupSchemaCache::rootPayloadFollows(const CacheGroupSchemaKey&         key,
                                               const std::vector<CacheGroupHint>& schema) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = schemas_.find(key);
    return it == schemas_.end() || !sameCacheGroupSchema(it->second, schema);
}

void CacheGroupSchemaCache::refresh(const CacheGroupSchemaKey& key, const std::vector<CacheGroupHint>& schema) {
    std::lock_guard<std::mutex> lock(mutex_);
    schemas_[key] = schema;
}

std::vector<CacheGroupHint> CacheGroupSchemaCache::lookup(const CacheGroupSchemaKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = schemas_.find(key);
    RTP_LLM_CHECK_WITH_INFO(
        it != schemas_.end(),
        "KV cache group schema missing after root announced no payload; TP communicator lifecycle is inconsistent "
        "(hash=%llu groups=%zu generation=%llu). Rebuild the communicator and all participating ranks together",
        static_cast<unsigned long long>(key.hash),
        key.group_count,
        static_cast<unsigned long long>(key.communicator_generation));
    return it->second;
}

std::vector<int32_t> encodeCacheGroupHintSchema(const std::vector<CacheGroupHint>& hints) {
    RTP_LLM_CHECK_WITH_INFO(hints.size() <= CacheGroupHintWireFormat::kMaxGroups,
                            "too many tagged KV cache groups: %zu > %zu",
                            hints.size(),
                            CacheGroupHintWireFormat::kMaxGroups);
    std::vector<int32_t> wire;
    for (const auto& hint : hints) {
        RTP_LLM_CHECK_WITH_INFO(hint.tag.size() <= CacheGroupHintWireFormat::kMaxTagBytes,
                                "KV cache tag is too long: tag=%s length=%zu",
                                hint.tag.c_str(),
                                hint.tag.size());
        const size_t tag_words = CacheGroupHintWireFormat::tagWords(hint.tag.size());
        const size_t offset    = wire.size();
        wire.resize(offset + CacheGroupHintWireFormat::kSchemaHeaderWords + tag_words, 0);
        wire[offset]     = static_cast<int32_t>(hint.tag.size());
        wire[offset + 1] = static_cast<int32_t>(hint.type);
        std::memcpy(
            wire.data() + offset + CacheGroupHintWireFormat::kSchemaHeaderWords, hint.tag.data(), hint.tag.size());
    }
    return wire;
}

std::vector<CacheGroupHint> decodeCacheGroupHints(const std::vector<int32_t>& schema_wire,
                                                  size_t                      expected_group_count,
                                                  const std::vector<int32_t>& widths) {
    RTP_LLM_CHECK_WITH_INFO(expected_group_count <= CacheGroupHintWireFormat::kMaxGroups,
                            "invalid broadcast KV cache group count: %zu > %zu",
                            expected_group_count,
                            CacheGroupHintWireFormat::kMaxGroups);
    std::vector<CacheGroupHint> hints;
    size_t                      offset = 0;
    hints.reserve(expected_group_count);
    for (size_t i = 0; i < expected_group_count; ++i) {
        RTP_LLM_CHECK_WITH_INFO(offset + CacheGroupHintWireFormat::kSchemaHeaderWords <= schema_wire.size(),
                                "truncated KV cache group schema header at group %zu",
                                i);
        const int32_t tag_size_raw = schema_wire[offset];
        const int32_t type_raw     = schema_wire[offset + 1];
        RTP_LLM_CHECK_WITH_INFO(tag_size_raw >= 0
                                    && static_cast<size_t>(tag_size_raw) <= CacheGroupHintWireFormat::kMaxTagBytes,
                                "invalid broadcast KV cache tag length=%d",
                                tag_size_raw);
        RTP_LLM_CHECK_WITH_INFO(type_raw >= static_cast<int32_t>(CacheGroupType::LINEAR)
                                    && type_raw <= static_cast<int32_t>(CacheGroupType::SWA),
                                "invalid broadcast KV cache group type=%d",
                                type_raw);
        const size_t tag_size  = static_cast<size_t>(tag_size_raw);
        const size_t tag_words = CacheGroupHintWireFormat::tagWords(tag_size);
        RTP_LLM_CHECK_WITH_INFO(offset + CacheGroupHintWireFormat::kSchemaHeaderWords + tag_words <= schema_wire.size(),
                                "truncated KV cache group tag at group %zu",
                                i);
        const char* tag_data =
            reinterpret_cast<const char*>(schema_wire.data() + offset + CacheGroupHintWireFormat::kSchemaHeaderWords);
        CacheGroupHint hint{std::string(tag_data, tag_size), static_cast<CacheGroupType>(type_raw)};
        const auto     duplicate =
            std::find_if(hints.begin(), hints.end(), [&](const auto& existing) { return existing.tag == hint.tag; });
        RTP_LLM_CHECK_WITH_INFO(duplicate == hints.end(), "duplicate broadcast KV cache tag=%s", hint.tag.c_str());
        hints.emplace_back(std::move(hint));
        offset += CacheGroupHintWireFormat::kSchemaHeaderWords + tag_words;
    }
    RTP_LLM_CHECK_WITH_INFO(
        offset == schema_wire.size(), "KV cache group schema has %zu trailing words", schema_wire.size() - offset);
    return applyCacheGroupHintWidths(hints, widths);
}

std::vector<CacheGroupHint> applyCacheGroupHintWidths(const std::vector<CacheGroupHint>& schema,
                                                      const std::vector<int32_t>&        widths) {
    RTP_LLM_CHECK_WITH_INFO(widths.size() == schema.size() * CacheGroupHintWireFormat::kWidthWordsPerGroup,
                            "invalid KV cache group width count: %zu for %zu groups",
                            widths.size(),
                            schema.size());
    auto hints = schema;
    for (size_t i = 0; i < hints.size(); ++i) {
        const int32_t block_width_raw  = widths[i * CacheGroupHintWireFormat::kWidthWordsPerGroup];
        const int32_t kernel_width_raw = widths[i * CacheGroupHintWireFormat::kWidthWordsPerGroup + 1];
        RTP_LLM_CHECK_WITH_INFO(block_width_raw >= 0 && kernel_width_raw >= 0,
                                "invalid KV cache group widths: physical=%d kernel=%d",
                                block_width_raw,
                                kernel_width_raw);
        hints[i].block_width        = static_cast<size_t>(block_width_raw);
        hints[i].kernel_block_width = static_cast<size_t>(kernel_width_raw);
    }
    return hints;
}

BlockTablesByGroup reconstructCacheGroupBlockTables(const std::vector<CacheGroupHint>& hints,
                                                    size_t                             batch_size,
                                                    const torch::Tensor&               physical_backing,
                                                    const torch::Tensor&               kernel_backing) {
    size_t expected_physical_numel = 0;
    size_t expected_kernel_numel   = 0;
    for (const auto& hint : hints) {
        expected_physical_numel += batch_size * hint.block_width;
        expected_kernel_numel += batch_size * hint.kernel_block_width;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(physical_backing.numel()) == expected_physical_numel,
                            "invalid physical block table backing size: %ld != %zu",
                            physical_backing.numel(),
                            expected_physical_numel);
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kernel_backing.numel()) == expected_kernel_numel,
                            "invalid kernel block table backing size: %ld != %zu",
                            kernel_backing.numel(),
                            expected_kernel_numel);

    BlockTablesByGroup tables;
    size_t             physical_offset = 0;
    size_t             kernel_offset   = 0;
    for (const auto& hint : hints) {
        GroupBlockTable table;
        table.tag       = hint.tag;
        table.type      = hint.type;
        table.block_ids = physical_backing.narrow(0, physical_offset, batch_size * hint.block_width)
                              .view({static_cast<int64_t>(batch_size), static_cast<int64_t>(hint.block_width)});
        table.kernel_block_ids =
            kernel_backing.narrow(0, kernel_offset, batch_size * hint.kernel_block_width)
                .view({static_cast<int64_t>(batch_size), static_cast<int64_t>(hint.kernel_block_width)});
        physical_offset += batch_size * hint.block_width;
        kernel_offset += batch_size * hint.kernel_block_width;
        const auto [it, inserted] = tables.emplace(hint.tag, std::move(table));
        (void)it;
        RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate broadcast KV cache tag=%s", hint.tag.c_str());
    }
    return tables;
}

void tpSyncModelInputs(GptModelInputs& inputs, const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }
    // The UDS-backed CPU broadcaster (used by execBroadcastCpu below) is
    // bootstrapped from Python in collective_torch._register_process_groups_to_cpp,
    // which guarantees deterministic timing across TP siblings. Cross-node TP
    // skips the init and falls back to NCCL automatically inside execBroadcastCpu.

    constexpr size_t kMaxCacheGroups  = CacheGroupHintWireFormat::kMaxGroups;
    constexpr size_t kMaxCacheTagSize = CacheGroupHintWireFormat::kMaxTagBytes;
    const bool       is_non_root      = parallelism_config.tp_rank != 0;
    const size_t     shape_hints_size = CacheGroupHintWireFormat::kShapeHintWords;
    auto             shape_hints_t    = torch::zeros({(int64_t)shape_hints_size}, torch::kInt32).pin_memory();
    auto             shape_hints_ptr  = shape_hints_t.data_ptr<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens] = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] =
        inputs.input_lengths.defined() ? inputs.input_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.defined() ? inputs.sequence_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.defined() ? inputs.prefix_lengths.numel() : 0;
    int32_t max_kernel_blocks_hint = 0;
    int32_t max_blocks_hint        = 0;
    RTP_LLM_CHECK_WITH_INFO(inputs.group_block_tables.size() <= kMaxCacheGroups,
                            "too many tagged KV cache groups: %zu > %zu",
                            inputs.group_block_tables.size(),
                            kMaxCacheGroups);
    std::vector<CacheGroupHint> root_group_hints;
    root_group_hints.reserve(inputs.group_block_tables.size());
    size_t group_index = 0;
    for (const auto& [tag, table] : inputs.group_block_tables) {
        RTP_LLM_CHECK_WITH_INFO(table.tag == tag, "block table key/tag mismatch for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            tag.size() <= kMaxCacheTagSize, "KV cache tag is too long: tag=%s length=%zu", tag.c_str(), tag.size());
        RTP_LLM_CHECK_WITH_INFO(table.block_ids.dim() == 2 && table.kernel_block_ids.dim() == 2,
                                "KV cache tables must be two-dimensional for tag=%s",
                                tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(table.block_ids.size(1) <= std::numeric_limits<int32_t>::max()
                                    && table.kernel_block_ids.size(1) <= std::numeric_limits<int32_t>::max(),
                                "KV cache table width exceeds wire range for tag=%s",
                                tag.c_str());
        max_blocks_hint        = std::max(max_blocks_hint, static_cast<int32_t>(table.block_ids.size(1)));
        max_kernel_blocks_hint = std::max(max_kernel_blocks_hint, static_cast<int32_t>(table.kernel_block_ids.size(1)));
        const size_t width_offset =
            GptModelInputIndex::gptModelInputLength + group_index * CacheGroupHintWireFormat::kWidthWordsPerGroup;
        shape_hints_ptr[width_offset]     = static_cast<int32_t>(table.block_ids.size(1));
        shape_hints_ptr[width_offset + 1] = static_cast<int32_t>(table.kernel_block_ids.size(1));
        root_group_hints.push_back(CacheGroupHint{tag,
                                                  table.type,
                                                  static_cast<size_t>(table.block_ids.size(1)),
                                                  static_cast<size_t>(table.kernel_block_ids.size(1))});
        ++group_index;
    }
    const auto root_schema_wire                                  = encodeCacheGroupHintSchema(root_group_hints);
    const auto root_schema_hash                                  = cacheGroupSchemaHash(root_schema_wire);
    const auto root_communicator_generation                      = cpuTpBroadcasterGeneration();
    shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch] = max_kernel_blocks_hint;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch]       = max_blocks_hint;
    shape_hints_ptr[GptModelInputIndex::cacheKeysWidth] =
        inputs.cache_keys.defined() && inputs.cache_keys.dim() >= 2 ? inputs.cache_keys.size(1) : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum] = inputs.group_block_tables.size();
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaGenerationLow] =
        static_cast<int32_t>(root_communicator_generation & std::numeric_limits<uint32_t>::max());
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaGenerationHigh] =
        static_cast<int32_t>(root_communicator_generation >> 32);
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaHashLow] =
        static_cast<int32_t>(root_schema_hash & std::numeric_limits<uint32_t>::max());
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaHashHigh]  = static_cast<int32_t>(root_schema_hash >> 32);
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaWireWords] = static_cast<int32_t>(root_schema_wire.size());
    const CacheGroupSchemaKey root_schema_key{root_schema_hash, root_group_hints.size(), root_communicator_generation};
    shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaPayloadFollows] =
        !is_non_root && !root_group_hints.empty()
        && processCacheGroupSchemaCache().rootPayloadFollows(root_schema_key, root_group_hints);
    // Kept as a reserved zero-valued slot for shape-hint wire compatibility.
    shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen] = 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen]   = 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum]   = inputs.kv_cache_update_mapping.size();
    shape_hints_ptr[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.defined() ? inputs.combo_position_ids.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::textTokensMask] =
        inputs.text_tokens_mask.defined() ? inputs.text_tokens_mask.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs] =
        inputs.mm_features_locs.defined() ? inputs.mm_features_locs.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] =
        inputs.multimodal_features.has_value() ? inputs.multimodal_features.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesSize] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0].size(1) : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ?
            (std::uint8_t)torchDTypeToDataType(inputs.multimodal_features.value()[0].dtype()) :
            0;
    shape_hints_ptr[GptModelInputIndex::mmHasExtraInput] =
        inputs.mm_extra_input.has_value() ? inputs.mm_extra_input.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmExtraInputDtype] =
        (inputs.mm_extra_input.has_value() && !inputs.mm_extra_input.value().empty()) ?
            (std::uint8_t)torchDTypeToDataType(inputs.mm_extra_input.value()[0].dtype()) :
            0;
    shape_hints_ptr[GptModelInputIndex::needAllLogits]       = inputs.need_all_logits;
    shape_hints_ptr[GptModelInputIndex::needAllHiddenStates] = inputs.need_all_hidden_states;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] =
        inputs.last_hidden_states.defined() ? inputs.last_hidden_states.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype] =
        inputs.last_hidden_states.defined() ? (std::uint8_t)torchDTypeToDataType(inputs.last_hidden_states.dtype()) : 0;
    shape_hints_ptr[GptModelInputIndex::skipRun] = inputs.skip_run;
    shape_hints_ptr[GptModelInputIndex::gptModelRequestLength] =
        inputs.request_id.defined() ? inputs.request_id.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::isFakeStream] = inputs.is_fake_stream;
    {
        // encode root-side tensor device for fields that may live on
        // GPU on the PDFUSION fast path, so non-root ranks can allocate matching
        // GPU buffers below and tpSync's pack/unpack stays in lockstep.
        uint32_t device_bits = 0;
        if (inputs.combo_tokens.defined() && inputs.combo_tokens.is_cuda()) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitComboTokens;
        }
        if (inputs.input_lengths.defined() && inputs.input_lengths.is_cuda()) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitInputLengths;
        }
        if (inputs.sequence_lengths.defined() && inputs.sequence_lengths.is_cuda()) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitSequenceLengths;
        }
        if (inputs.prefix_lengths.defined() && inputs.prefix_lengths.is_cuda()) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitPrefixLengths;
        }
        if (inputs.lm_output_indexes.defined() && inputs.lm_output_indexes.is_cuda()) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitLmOutputIndexes;
        }
        const bool kernel_block_ids_on_cuda =
            std::any_of(inputs.group_block_tables.begin(), inputs.group_block_tables.end(), [](const auto& item) {
                return item.second.kernel_block_ids.defined() && item.second.kernel_block_ids.is_cuda();
            });
        if (kernel_block_ids_on_cuda) {
            device_bits |= GptModelInputDeviceBit::kDeviceBitKernelBlockId;
        }
        shape_hints_ptr[GptModelInputIndex::tensorDeviceMap] = static_cast<int32_t>(device_bits);
    }

    // CPU broadcast: routed through CpuTpBroadcaster (UDS) when intra-node;
    // execBroadcastCpu's fallback path keeps the NCCL+cudaSyncAndCheck
    // contract for cross-node TP.
    execBroadcastCpu({{shape_hints_t}, 0});

    // multimodal features shape broadcast
    torch::Tensor mm_features_shape_t;
    int32_t*      mm_features_shape_ptr = nullptr;
    // extra-input (model-specific, treated as opaque flat 1-D tensors) per-tensor element count
    torch::Tensor mm_extra_input_shape_t;
    int64_t*      mm_extra_input_shape_ptr = nullptr;
    inputs.need_all_logits                 = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.need_all_hidden_states          = shape_hints_ptr[GptModelInputIndex::needAllHiddenStates];
    inputs.skip_run                        = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream                  = shape_hints_ptr[GptModelInputIndex::isFakeStream];
    if (inputs.skip_run) {
        return;
    }

    const size_t kv_cache_group_num = static_cast<size_t>(shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum]);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_group_num <= kMaxCacheGroups,
                            "invalid broadcast KV cache group count: %zu > %zu",
                            kv_cache_group_num,
                            kMaxCacheGroups);
    const uint64_t schema_hash =
        static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaHashLow])
        | (static_cast<uint64_t>(static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaHashHigh]))
           << 32);
    const uint64_t communicator_generation =
        static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaGenerationLow])
        | (static_cast<uint64_t>(
               static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaGenerationHigh]))
           << 32);
    const size_t schema_wire_words =
        static_cast<size_t>(shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaWireWords]);
    RTP_LLM_CHECK_WITH_INFO(schema_wire_words <= CacheGroupHintWireFormat::kMaxSchemaWords,
                            "invalid KV cache group schema size: %zu > %zu",
                            schema_wire_words,
                            CacheGroupHintWireFormat::kMaxSchemaWords);
    const CacheGroupSchemaKey schema_key{schema_hash, kv_cache_group_num, communicator_generation};
    const bool schema_payload_follows = shape_hints_ptr[GptModelInputIndex::cacheGroupSchemaPayloadFollows] != 0;
    if (schema_payload_follows) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_group_num > 0, "root announced a KV cache schema payload for an empty schema");
        auto schema_t = torch::zeros({static_cast<int64_t>(schema_wire_words)}, torch::kInt32).pin_memory();
        if (!is_non_root) {
            RTP_LLM_CHECK_WITH_INFO(root_schema_wire.size() == schema_wire_words,
                                    "KV cache group schema changed during TP sync");
            std::memcpy(schema_t.data_ptr<int32_t>(), root_schema_wire.data(), schema_wire_words * sizeof(int32_t));
        }
        // Tags and types are static for a model topology. Only the first use of
        // a schema takes this compact broadcast; steady-state steps use the
        // cached schema and carry dynamic widths in shape_hints_t.
        execBroadcastCpu({{schema_t}, 0});
        std::vector<int32_t> schema_wire(schema_t.data_ptr<int32_t>(),
                                         schema_t.data_ptr<int32_t>() + schema_wire_words);
        std::vector<int32_t> zero_widths(kv_cache_group_num * CacheGroupHintWireFormat::kWidthWordsPerGroup, 0);
        auto                 hints = decodeCacheGroupHints(schema_wire, kv_cache_group_num, zero_widths);
        processCacheGroupSchemaCache().refresh(schema_key, hints);
    }

    std::vector<int32_t> group_width_wire(kv_cache_group_num * CacheGroupHintWireFormat::kWidthWordsPerGroup);
    std::copy_n(
        shape_hints_ptr + GptModelInputIndex::gptModelInputLength, group_width_wire.size(), group_width_wire.begin());
    const auto group_hints =
        kv_cache_group_num == 0 ?
            std::vector<CacheGroupHint>{} :
            applyCacheGroupHintWidths(processCacheGroupSchemaCache().lookup(schema_key), group_width_wire);

    const size_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    if (mm_features_num) {
        mm_features_shape_t   = torch::empty({(int64_t)mm_features_num}, torch::kInt32).pin_memory();
        mm_features_shape_ptr = mm_features_shape_t.data_ptr<int32_t>();
        for (size_t i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] =
                inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i].size(0) : 0;
        }
        // CPU broadcast (UDS path; fallback handles cudaSyncAndCheck).
        execBroadcastCpu({{mm_features_shape_t}, 0});
    }

    // extra-input element counts broadcast: each extra-input is an opaque flat 1-D tensor,
    // so we send its element count first ("先传shape") and allocate a 1-D buffer on non-root.
    const size_t mm_extra_input_num = (size_t)shape_hints_ptr[GptModelInputIndex::mmHasExtraInput];
    if (mm_extra_input_num) {
        mm_extra_input_shape_t   = torch::empty({(int64_t)mm_extra_input_num}, torch::kInt64).pin_memory();
        mm_extra_input_shape_ptr = mm_extra_input_shape_t.data_ptr<int64_t>();
        for (size_t i = 0; i < mm_extra_input_num; ++i) {
            mm_extra_input_shape_ptr[i] =
                inputs.mm_extra_input.has_value() ? inputs.mm_extra_input.value()[i].numel() : 0;
        }
        execBroadcast({{mm_extra_input_shape_t}, 0});
        execSyncCommunication(false);
        cudaSyncAndCheck();
    }

    auto   max_kernel_blocks       = (size_t)shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch];
    auto   max_blocks              = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto   cache_keys_width        = (size_t)shape_hints_ptr[GptModelInputIndex::cacheKeysWidth];
    auto   combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto   text_tokens_mask_size   = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto   mm_features_locs_size   = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];
    auto   hidden_states_size      = shape_hints_ptr[GptModelInputIndex::mtpHiddenStates];
    size_t request_length          = shape_hints_ptr[GptModelInputIndex::gptModelRequestLength];

    std::map<std::string, CacheGroupType> group_types;
    std::map<std::string, size_t>         group_widths;
    std::map<std::string, size_t>         group_kernel_widths;
    for (const auto& hint : group_hints) {
        const bool type_inserted         = group_types.emplace(hint.tag, hint.type).second;
        const bool width_inserted        = group_widths.emplace(hint.tag, hint.block_width).second;
        const bool kernel_width_inserted = group_kernel_widths.emplace(hint.tag, hint.kernel_block_width).second;
        RTP_LLM_CHECK_WITH_INFO(type_inserted && width_inserted && kernel_width_inserted,
                                "duplicate broadcast KV cache tag=%s",
                                hint.tag.c_str());
    }

    auto allocBuf = [&](rtp_llm::DataType       dtype,
                        std::vector<size_t>     dims,
                        rtp_llm::AllocationType atype = rtp_llm::AllocationType::HOST) -> torch::Tensor {
        auto torch_dtype = dataTypeToTorchType(dtype);
        auto options     = torch::TensorOptions(torch_dtype);
        if (atype == rtp_llm::AllocationType::DEVICE) {
            options = options.device(torch::kCUDA);
        }
        std::vector<int64_t> dims64(dims.begin(), dims.end());
        auto                 tensor = torch::empty(dims64, options);
        // NCCL broadcast requires pinned memory for CPU buffers
        if (atype != rtp_llm::AllocationType::DEVICE) {
            tensor = tensor.pin_memory();
        }
        return tensor;
    };

    if (is_non_root) {
        auto context_batch_size = (size_t)shape_hints_ptr[GptModelInputIndex::prefixLengths];

        // Respect the root-side device bitmap so all ranks classify tensors the
        // same way and preserve NCCL broadcast ordering.
        const uint32_t device_bits = static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::tensorDeviceMap]);
        auto           pickAlloc   = [&](GptModelInputDeviceBit bit) {
            return (device_bits & bit) ? rtp_llm::AllocationType::DEVICE : rtp_llm::AllocationType::HOST;
        };

        inputs.combo_tokens     = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {(size_t)shape_hints_ptr[GptModelInputIndex::comboTokens]},
                                       pickAlloc(GptModelInputDeviceBit::kDeviceBitComboTokens));
        inputs.input_lengths    = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths]},
                                        pickAlloc(GptModelInputDeviceBit::kDeviceBitInputLengths));
        inputs.sequence_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {(size_t)shape_hints_ptr[GptModelInputIndex::sequenceLengths]},
                                           pickAlloc(GptModelInputDeviceBit::kDeviceBitSequenceLengths));
        inputs.prefix_lengths   = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {context_batch_size},
                                         pickAlloc(GptModelInputDeviceBit::kDeviceBitPrefixLengths));
        if (max_kernel_blocks != 0 || max_blocks != 0) {
            const size_t batch_size     = shape_hints_ptr[GptModelInputIndex::inputLengths];
            size_t       physical_numel = 0;
            size_t       kernel_numel   = 0;
            for (const auto& [tag, type] : group_types) {
                (void)type;
                physical_numel += batch_size * group_widths.at(tag);
                kernel_numel += batch_size * group_kernel_widths.at(tag);
            }
            auto physical_backing = allocBuf(rtp_llm::DataType::TYPE_INT32, {physical_numel});
            auto kernel_backing   = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                             {kernel_numel},
                                           pickAlloc(GptModelInputDeviceBit::kDeviceBitKernelBlockId));
            inputs.group_block_tables =
                reconstructCacheGroupBlockTables(group_hints, batch_size, physical_backing, kernel_backing);
            if (inputs.pd_separation) {
                inputs.cache_keys = allocBuf(rtp_llm::DataType::TYPE_INT64,
                                             {context_batch_size, cache_keys_width ? cache_keys_width : max_blocks});
            }
        }
        inputs.request_id            = allocBuf(rtp_llm::DataType::TYPE_INT64, {request_length});
        inputs.request_pd_separation = allocBuf(rtp_llm::DataType::TYPE_BOOL, {request_length});
        inputs.lm_output_indexes     = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputIndexes]},
                                            pickAlloc(GptModelInputDeviceBit::kDeviceBitLmOutputIndexes));
        if (combo_position_ids_size) {
            inputs.combo_position_ids = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)combo_position_ids_size});
        }
        if (shape_hints_ptr[GptModelInputIndex::mtpHiddenStates]) {
            auto hidden_states_dim0 = (size_t)shape_hints_ptr[GptModelInputIndex::comboTokens];
            auto hidden_states_dim1 = (size_t)hidden_states_size / hidden_states_dim0;
            RTP_LLM_CHECK(hidden_states_size % hidden_states_dim0 == 0);
            inputs.last_hidden_states =
                allocBuf((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype],
                         {hidden_states_dim0, hidden_states_dim1},
                         rtp_llm::AllocationType::DEVICE);
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)text_tokens_mask_size});
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)mm_features_locs_size});
        }
        if (mm_features_num) {
            std::vector<torch::Tensor> mm_features;
            auto                       mm_dtype =
                dataTypeToTorchType((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype]);
            for (auto mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                mm_features.emplace_back(torch::empty({(int64_t)mm_features_shape_ptr[mm_index],
                                                       (int64_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesSize]},
                                                      torch::TensorOptions().dtype(mm_dtype).device(torch::kCUDA)));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
        if (mm_extra_input_num) {
            std::vector<torch::Tensor> mm_extra_input;
            auto                       extra_dtype =
                dataTypeToTorchType((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmExtraInputDtype]);
            for (size_t i = 0; i < mm_extra_input_num; ++i) {
                mm_extra_input.emplace_back(
                    torch::empty({(int64_t)mm_extra_input_shape_ptr[i]},
                                 torch::TensorOptions().dtype(extra_dtype).device(torch::kCUDA)));
            }
            inputs.mm_extra_input = std::move(mm_extra_input);
        }
    }

    constexpr size_t kUpdateWireRecordSize = kMaxCacheTagSize + 2 * sizeof(int32_t);
    const size_t     update_count = static_cast<size_t>(shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum]);
    torch::Tensor    update_wire;
    if (update_count > 0) {
        update_wire = torch::zeros({static_cast<int64_t>(update_count * kUpdateWireRecordSize)},
                                   torch::TensorOptions(torch::kUInt8))
                          .pin_memory();
        if (!is_non_root) {
            RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_update_mapping.size() == update_count,
                                    "KV cache update mapping count changed during TP sync");
            auto* wire = update_wire.data_ptr<uint8_t>();
            for (size_t i = 0; i < update_count; ++i) {
                const auto& mapping = inputs.kv_cache_update_mapping[i];
                RTP_LLM_CHECK_WITH_INFO(mapping.tag.size() <= kMaxCacheTagSize,
                                        "KV cache update tag is too long: tag=%s",
                                        mapping.tag.c_str());
                auto* record = wire + i * kUpdateWireRecordSize;
                std::memcpy(record, mapping.tag.data(), mapping.tag.size());
                std::memcpy(record + kMaxCacheTagSize, &mapping.src, sizeof(mapping.src));
                std::memcpy(record + kMaxCacheTagSize + sizeof(mapping.src), &mapping.dst, sizeof(mapping.dst));
            }
        }
    }

    // Collect all tensors that participate in broadcast.
    // The collect order must be deterministic and identical across all ranks.
    std::vector<torch::Tensor*> tensor_ptrs;
    auto                        collect = [&](torch::Tensor& t) {
        if (t.defined() && t.numel() > 0) {
            tensor_ptrs.push_back(&t);
        }
    };

    collect(inputs.combo_tokens);
    collect(inputs.input_lengths);
    collect(inputs.sequence_lengths);
    collect(inputs.prefix_lengths);
    if (max_kernel_blocks || max_blocks) {
        for (auto& [tag, table] : inputs.group_block_tables) {
            (void)tag;
            collect(table.kernel_block_ids);
            collect(table.block_ids);
        }
        if (inputs.pd_separation) {
            collect(inputs.cache_keys);
        }
    }
    if (update_wire.defined()) {
        collect(update_wire);
    }
    collect(inputs.request_id);
    collect(inputs.request_pd_separation);
    collect(inputs.lm_output_indexes);
    if (combo_position_ids_size) {
        collect(inputs.combo_position_ids);
    }
    if (text_tokens_mask_size) {
        collect(inputs.text_tokens_mask);
    }
    if (mm_features_locs_size) {
        collect(inputs.mm_features_locs);
    }
    if (mm_features_num) {
        for (auto& f : inputs.multimodal_features.value()) {
            collect(f);
        }
    }
    if (mm_extra_input_num) {
        for (auto& e : inputs.mm_extra_input.value()) {
            collect(e);
        }
    }
    if (hidden_states_size) {
        collect(inputs.last_hidden_states);
    }

    // Classify tensors by device type (runtime check) and calculate packed sizes.
    // Align each entry to 16 bytes so that typed access at any offset is safe
    // and GPU memory coalescing / NCCL transfers stay on fast paths.
    constexpr int64_t kPackAlignment = 16;
    auto              align_up       = [](int64_t size, int64_t alignment) -> int64_t {
        return (size + alignment - 1) & ~(alignment - 1);
    };

    struct PackEntry {
        torch::Tensor* tensor;
        int64_t        offset;
        int64_t        nbytes;
    };
    std::vector<PackEntry> cpu_entries, gpu_entries;
    int64_t                cpu_total_bytes = 0, gpu_total_bytes = 0;

    for (auto* tp : tensor_ptrs) {
        auto nb = static_cast<int64_t>(tp->nbytes());
        if (tp->is_cuda()) {
            gpu_entries.push_back({tp, gpu_total_bytes, nb});
            gpu_total_bytes += align_up(nb, kPackAlignment);
        } else {
            cpu_entries.push_back({tp, cpu_total_bytes, nb});
            cpu_total_bytes += align_up(nb, kPackAlignment);
        }
    }

    bool is_root = parallelism_config.tp_rank == 0;

    // Allocate one packed buffer per device type.
    // CPU buffer uses pinned memory (required by NCCL for host-side broadcast).
    torch::Tensor cpu_packed, gpu_packed;

    if (cpu_total_bytes > 0) {
        cpu_packed = torch::empty({cpu_total_bytes}, torch::kUInt8).pin_memory();
        if (is_root) {
            auto* base = static_cast<uint8_t*>(cpu_packed.data_ptr());
            for (auto& e : cpu_entries) {
                auto contig = e.tensor->contiguous();
                std::memcpy(base + e.offset, contig.data_ptr(), e.nbytes);
            }
        }
    }

    if (gpu_total_bytes > 0) {
        gpu_packed = torch::empty({gpu_total_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        if (is_root) {
            auto*              packed_base = static_cast<uint8_t*>(gpu_packed.data_ptr());
            FusedD2DCopyParams fused_params;
            auto               flush_fused_copy = [&]() {
                if (fused_params.num_copies > 0) {
                    fusedCopy(fused_params);
                    fused_params.clear();
                }
            };
            for (auto& e : gpu_entries) {
                if (e.tensor->is_contiguous()) {
                    if (fused_params.num_copies == MAX_FUSED_D2D_COPIES) {
                        flush_fused_copy();
                    }
                    fused_params.add(e.tensor->data_ptr(), packed_base + e.offset, static_cast<size_t>(e.nbytes));
                    continue;
                }

                // Preserve the old logical-order copy for rare non-contiguous tensors.
                flush_fused_copy();
                auto contig    = e.tensor->contiguous();
                auto src_bytes = torch::from_blob(
                    contig.data_ptr(), {e.nbytes}, torch::TensorOptions(torch::kUInt8).device(contig.device()));
                gpu_packed.narrow(0, e.offset, e.nbytes).copy_(src_bytes);
            }
            flush_fused_copy();
        }
    }

    // Broadcast at most 2 packed buffers instead of N individual tensors.
    if (cpu_packed.defined()) {
        execBroadcastCpu({{cpu_packed}, 0});
    }

    if (gpu_packed.defined()) {
        // gpu no need to sync communication
        execBroadcast({{gpu_packed}, 0});
    }

    // Unpack from packed buffers back to each tensor's original storage.
    if (!is_root) {
        if (cpu_total_bytes > 0) {
            auto* base = static_cast<const uint8_t*>(cpu_packed.data_ptr());
            for (auto& e : cpu_entries) {
                std::memcpy(e.tensor->data_ptr(), base + e.offset, e.nbytes);
            }
        }
        if (gpu_total_bytes > 0) {
            auto*              packed_base = static_cast<uint8_t*>(gpu_packed.data_ptr());
            FusedD2DCopyParams fused_params;
            auto               flush_fused_copy = [&]() {
                if (fused_params.num_copies > 0) {
                    fusedCopy(fused_params);
                    fused_params.clear();
                }
            };
            for (auto& e : gpu_entries) {
                if (e.tensor->is_contiguous()) {
                    if (fused_params.num_copies == MAX_FUSED_D2D_COPIES) {
                        flush_fused_copy();
                    }
                    fused_params.add(packed_base + e.offset, e.tensor->data_ptr(), static_cast<size_t>(e.nbytes));
                    continue;
                }

                flush_fused_copy();
                auto src_tensor = torch::from_blob(packed_base + e.offset, e.tensor->sizes(), e.tensor->options());
                e.tensor->copy_(src_tensor);
            }
            flush_fused_copy();
        }
    }

    if (is_non_root) {
        inputs.kv_cache_update_mapping.clear();
        inputs.kv_cache_update_mapping.reserve(update_count);
        const auto* wire = update_wire.defined() ? update_wire.data_ptr<uint8_t>() : nullptr;
        for (size_t i = 0; i < update_count; ++i) {
            const auto*      record   = wire + i * kUpdateWireRecordSize;
            const auto*      end      = static_cast<const uint8_t*>(std::memchr(record, '\0', kMaxCacheTagSize));
            const size_t     tag_size = end ? static_cast<size_t>(end - record) : kMaxCacheTagSize;
            GroupBlockIdPair mapping;
            mapping.tag.assign(reinterpret_cast<const char*>(record), tag_size);
            std::memcpy(&mapping.src, record + kMaxCacheTagSize, sizeof(mapping.src));
            std::memcpy(&mapping.dst, record + kMaxCacheTagSize + sizeof(mapping.src), sizeof(mapping.dst));
            inputs.kv_cache_update_mapping.push_back(std::move(mapping));
        }
    }
}

}  // namespace rtp_llm
