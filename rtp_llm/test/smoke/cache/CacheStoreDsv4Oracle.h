#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <numeric>
#include <thread>
#include <unistd.h>

#include "rtp_llm/test/smoke/cache/CacheSmokeSupport.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::cache_smoke {

inline std::set<std::string> dsvExpectedTags(int ratio) {
    if (ratio == 4)
        return {"csa_kv", "indexer_kv", "indexer_state", "csa_state", "swa_kv"};
    if (ratio == 128)
        return {"hca_kv", "hca_state", "swa_kv"};
    return {"swa_kv"};
}
// Oracle geometry is independent of native spec sizes and partition output.
struct Dsv4Unit {
    std::string         component;
    int                 global_head = 0;
    std::vector<size_t> offsets;
};
struct Dsv4Oracle {
    uint64_t seed;
    int      block_tokens, head_dim, indexer_dim;
    bool     fixed_host;
    bool     cp_enabled = false, sliced = false, page_sharded = false;
    int      rank = 0;
    bool     fixed(const std::string& tag) const {
        return tag == "swa_kv" || tag.find("state") != std::string::npos;
    }
    size_t global(const std::string& tag, size_t local) const {
        return !cp_enabled ? local : fixed(tag) ? local * 2 + 1 : page_sharded ? local * 2 + rank : local;
    }
    bool host(const std::string& tag) const {
        return fixed_host && (tag == "swa_kv" || tag.find("state") != std::string::npos);
    }
    size_t entry(const std::string& tag) const {
        if (tag == "indexer_kv")
            return 132;
        if (tag == "indexer_state")
            return 16 * indexer_dim;
        if (tag == "csa_state")
            return 16 * head_dim;
        if (tag == "hca_state")
            return 8 * head_dim;
        return 584;
    }
    size_t entries(const std::string& tag) const {
        if (tag == "csa_kv" || tag == "indexer_kv")
            return block_tokens / 4;
        if (tag == "hca_kv")
            return block_tokens / 128;
        if (tag == "csa_state" || tag == "indexer_state")
            return 8;
        return 128;  // Non-MTP HCA/SWA production fixed ring.
    }
    size_t fullPayload(const std::string& tag) const {
        return entry(tag) * entries(tag);
    }
    size_t fullStride(const std::string& tag) const {
        const size_t bytes = fullPayload(tag);
        return tag == "csa_kv" || tag == "hca_kv" || tag == "swa_kv" ? (bytes + 575) / 576 * 576 : bytes;
    }
    size_t payloadBytes(const std::string& tag) const {
        return fullPayload(tag) / (sliced && fixed(tag) && tag != "swa_kv" ? 2 : 1);
    }
    size_t stride(const std::string& tag) const {
        return fullStride(tag) / (sliced && fixed(tag) ? 2 : 1);
    }

    std::vector<Dsv4Unit> units(const std::string& tag) const {
        std::vector<Dsv4Unit> result;
        const std::string     component = tag.find("state") != std::string::npos ? "FP32_state" :
                                          tag == "indexer_kv"                    ? "FP8_indexer_packed" :
                                                                                   "FP8_KV_packed";
        const size_t          base      = sliced && fixed(tag) ? rank * stride(tag) : 0;
        const int             parts     = cp_enabled && fixed(tag) ? 2 : 1;
        for (int part = 0; part < parts; ++part) {
            if (sliced && fixed(tag) && part != rank)
                continue;
            const size_t begin = fullStride(tag) / parts * part, end = fullStride(tag) / parts * (part + 1);
            for (const auto& kind : {component, std::string("stride_padding")}) {
                Dsv4Unit unit{kind + (parts == 2 ? ".cp" + std::to_string(part) : ""), 0, {}};
                for (size_t i = begin; i < end; ++i)
                    if ((i < fullPayload(tag)) == (kind == component))
                        unit.offsets.push_back(i - base);
                if (!unit.offsets.empty())
                    result.push_back(std::move(unit));
            }
        }
        return result;
    }
    std::vector<uint8_t> payload(int layer, const std::string& tag, size_t logical) const {
        std::vector<uint8_t> bytes(fullStride(tag));
        const uint64_t       tag_id = signature(reinterpret_cast<const uint8_t*>(tag.data()), tag.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            uint64_t state = seed ^ (static_cast<uint64_t>(layer + 1) * UINT64_C(0xd6e8feb86659fd93));
            state ^= tag_id;
            state ^= (logical + 1) * UINT64_C(0x8ebc6af09c88c6e3);
            // Byte coordinates include all fixed FP8 packing/scales and stride padding;
            // no rank or physical block number enters replicated MQA contents.
            state ^= (i + 1) * UINT64_C(0x589965cc75374cc3);
            state    = (state ^ (state >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            state    = (state ^ (state >> 27)) * UINT64_C(0x94d049bb133111eb);
            bytes[i] = static_cast<uint8_t>(state ^ (state >> 31));
        }
        if (sliced && fixed(tag)) {
            const size_t offset = rank * stride(tag);
            return {bytes.begin() + offset, bytes.begin() + offset + stride(tag)};
        }
        return bytes;
    }
};
inline std::vector<uint8_t> dsvRead(void* address, size_t bytes, bool host) {
    if (!host)
        return readBytes(address, bytes);
    std::vector<uint8_t> result(bytes);
    std::memcpy(result.data(), address, bytes);
    return result;
}
inline void dsvWrite(void* address, const std::vector<uint8_t>& bytes, bool host) {
    if (!host) {
        writeBytes(address, bytes);
        return;
    }
    std::memcpy(address, bytes.data(), bytes.size());
}

// Parse a transport-free projection of the original production Python descs.
// Native SpecBuilder resolves all actual group policies and memory layouts.
inline void projectDsv4Descs(ModelConfig& model, const std::string& projection) {
    model.kv_cache_spec_descs.resize(model.num_layers);
    std::istringstream lines(projection);
    for (std::string line; std::getline(lines, line);) {
        std::vector<std::string> fields;
        std::istringstream       row(line);
        for (std::string field; std::getline(row, field, '|');)
            fields.push_back(field);
        require(fields.size() == 22, "DSV4 descriptor projection field count");
        const auto      number = [&](size_t i) { return std::stoi(fields.at(i)); };
        KVCacheSpecDesc d;
        d.tag         = fields[1];
        d.cache_type  = static_cast<KVCacheSpecType>(number(2));
        d.entry_elems = number(3);
        d.dtype = d.entry_dtype                = fields[4] == "uint8" ? DataType::TYPE_UINT8 : DataType::TYPE_FP32;
        d.is_state_cache                       = number(5);
        d.entry_count_mode                     = static_cast<OpaqueBlockEntryCountMode>(number(6));
        d.compression_ratio                    = number(7);
        d.state_ring_overlap                   = number(8);
        d.state_ring_include_gen_num_per_cycle = number(9);
        d.block_stride_bytes_alignment         = number(10);
        d.block_stride_alignment_min_entries   = number(11);
        if (number(12) >= 0) {
            CacheReusePolicyDesc v;
            v.enable_prefix_reuse = number(12);
            d.reuse               = v;
        }
        if (number(13) >= 0 || number(14) >= 0) {
            CacheCapacityPolicyDesc v;
            if (number(13) >= 0)
                v.explicit_block_num = number(13);
            if (number(14) >= 0)
                v.charge_to_paged_budget = number(14);
            d.capacity = v;
        }
        if (number(15) >= 0) {
            CacheMemoryPolicyDesc v;
            v.placement = static_cast<CacheMemoryPlacement>(number(15));
            d.memory    = v;
        }
        if (number(16) >= 0 || number(17) >= 0) {
            CacheTailPolicyDesc v;
            if (number(16) >= 0)
                v.active_tail_blocks = number(16);
            if (number(17) >= 0)
                v.validate_tail_blocks = number(17);
            d.tail = v;
        }
        if (number(18) >= 0) {
            CacheCpPolicyDesc v;
            v.slice                = static_cast<CpBlockSliceMode>(number(18));
            v.scale_seq_size       = number(19);
            v.align_payload        = number(20);
            v.prefill_slice_layout = static_cast<CpPrefillSliceLayout>(number(21));
            d.cp                   = v;
        }
        model.kv_cache_spec_descs.at(number(0)).push_back(d);
    }
}

inline CheckResult checkDsv4(KVCacheManager& manager, const KVCacheResource& resource, const Dsv4Oracle& oracle) {
    CheckResult result;
    for (int layer = 0; layer < resource.layerNum(); ++layer) {
        for (const auto& tag : resource.groupTagsForLayer(layer)) {
            const auto& blocks = resource.blocks(tag);
            for (size_t logical = 0; logical < blocks.size(); ++logical) {
                if (isNullBlockIdx(blocks[logical]))
                    continue;
                const auto expected = oracle.payload(layer, tag, oracle.global(tag, logical));
                const auto actual   = dsvRead(
                    manager.convertIndexToAddr(layer, tag, blocks[logical]).kv_addr, expected.size(), oracle.host(tag));
                result.matches &= actual == expected;
                result.expected = signature(expected.data(), expected.size(), result.expected);
                result.observed = signature(actual.data(), actual.size(), result.observed);
                result.bytes += actual.size();
                const std::string identity = "\"layer\":" + std::to_string(layer) + ",\"group\":" + quote(tag)
                                             + ",\"logical_block\":" + std::to_string(oracle.global(tag, logical))
                                             + ",\"local_ordinal\":" + std::to_string(logical);
                if (result.detail.size() > 1)
                    result.detail += ',';
                result.detail += "{" + identity + ",\"physical_block\":" + std::to_string(blocks[logical])
                                 + ",\"bytes\":" + std::to_string(actual.size())
                                 + ",\"expected\":" + quote(hex(signature(expected.data(), expected.size())))
                                 + ",\"observed\":" + quote(hex(signature(actual.data(), actual.size())))
                                 + ",\"matches\":" + (actual == expected ? "true" : "false") + "}";
                for (const auto& unit : oracle.units(tag)) {
                    std::vector<uint8_t> exp, obs;
                    for (size_t offset : unit.offsets) {
                        exp.push_back(expected[offset]);
                        obs.push_back(actual[offset]);
                    }
                    if (result.units.size() > 1)
                        result.units += ',';
                    result.units += "{" + identity + ",\"global_head\":" + std::to_string(unit.global_head)
                                    + ",\"component\":" + quote(unit.component)
                                    + ",\"scale\":false,\"bytes\":" + std::to_string(exp.size())
                                    + ",\"expected\":" + quote(hex(signature(exp.data(), exp.size())))
                                    + ",\"observed\":" + quote(hex(signature(obs.data(), obs.size())))
                                    + ",\"matches\":" + (exp == obs ? "true" : "false") + "}";
                    if (exp != obs) {
                        const size_t first = std::mismatch(exp.begin(), exp.end(), obs.begin()).first - exp.begin();
                        std::cerr << "DSV4 payload mismatch layer=" << layer << " group=" << tag
                                  << " logical_block=" << oracle.global(tag, logical) << " component=" << unit.component
                                  << " global_head=" << unit.global_head << " component_byte=" << first
                                  << " packed_byte=" << unit.offsets[first] << '\n';
                    }
                }
            }
        }
    }
    result.detail += ']';
    result.units += ']';
    return result;
}

inline bool dsvGuards(KVCacheManager& manager, const KVCacheResource& resource, const Dsv4Oracle& oracle) {
    for (const auto& group : manager.cacheConfig().groups()) {
        const auto&         blocks = resource.blocks(group.tag);
        const std::set<int> live(blocks.begin(), blocks.end());
        for (int layer : manager.cacheConfig().layerIdsForGroup(group.tag)) {
            for (uint32_t block = 0; block < group.block_num; ++block) {
                if (live.count(block))
                    continue;
                const auto bytes = dsvRead(manager.convertIndexToAddr(layer, group.tag, block).kv_addr,
                                           oracle.stride(group.tag),
                                           oracle.host(group.tag));
                if (!std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0xa5; }))
                    return false;
            }
        }
    }
    return true;
}

inline std::pair<BatchKVCacheResourcePtr, CompleteTokenIdsPtr>
dsvAllocate(KVCacheManager& manager, int tokens, int block_tokens, int64_t request) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(manager.cacheConfig().topologyPtr());
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::arange(tokens, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    auto complete          = std::make_shared<CompleteTokenIds>(1, 1, tokens + block_tokens, block_tokens);
    complete->init(input);
    MallocInfo allocation{resource, complete};
    allocation.request_id          = request;
    allocation.reuse_cache         = false;
    allocation.enable_cache_lookup = false;
    const auto allocated           = manager.malloc(allocation);
    require(allocated.success,
            "production DSV4 malloc failed status=" + std::to_string(static_cast<int>(allocated.status)));
    return {resource, complete};
}

}  // namespace rtp_llm::cache_smoke
