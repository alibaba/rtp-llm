#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <numeric>
#include <thread>
#include <unistd.h>

#include "rtp_llm/test/smoke/cache/CacheSmokeSupport.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::cache_smoke {

struct NextUnit {
    std::string         component;
    int                 global_head;
    std::vector<size_t> offsets;
};

struct NextOracle {
    uint64_t seed;
    int      tp_size, tp_rank, kv_heads, head_dim, block_tokens, key_heads, value_heads, linear_dim, conv_kernel;
    size_t   ssm_width, conv_width;

    int fullHeads() const {
        return kv_heads / std::gcd(kv_heads, tp_size);
    }
    int fullFirst() const {
        return tp_rank / std::max(1, tp_size / kv_heads) * fullHeads();
    }
    size_t ssmBytes() const {
        return value_heads / tp_size * linear_dim * linear_dim * ssm_width;
    }
    size_t convBytes() const {
        return (2 * key_heads + value_heads) / tp_size * linear_dim * (conv_kernel - 1) * conv_width;
    }

    std::vector<NextUnit> units(const std::string& tag) const {
        std::vector<NextUnit> result;
        if (tag == "full") {
            const size_t head_bytes = block_tokens * head_dim * 2;
            for (int c = 0; c < 2; ++c) {
                for (int h = 0; h < fullHeads(); ++h) {
                    NextUnit unit{c == 0 ? "K" : "V", fullFirst() + h, {}};
                    for (size_t i = 0; i < head_bytes; ++i)
                        unit.offsets.push_back((c * fullHeads() + h) * head_bytes + i);
                    result.push_back(std::move(unit));
                }
            }
        } else {
            const size_t head_bytes = linear_dim * linear_dim * ssm_width;
            for (int h = 0; h < value_heads / tp_size; ++h) {
                NextUnit unit{"SSM", tp_rank * value_heads / tp_size + h, {}};
                for (size_t i = 0; i < head_bytes; ++i)
                    unit.offsets.push_back(h * head_bytes + i);
                result.push_back(std::move(unit));
            }
            const int qkv          = (2 * key_heads + value_heads) / tp_size * linear_dim;
            int       channel_base = 0;
            for (const auto& component : {"conv_Q", "conv_K", "conv_V"}) {
                const int count = std::string(component) == "conv_V" ? value_heads : key_heads;
                for (int h = 0; h < count / tp_size; ++h) {
                    NextUnit unit{component, tp_rank * count / tp_size + h, {}};
                    for (int step = 0; step < conv_kernel - 1; ++step) {
                        for (int dim = 0; dim < linear_dim; ++dim) {
                            for (size_t lane = 0; lane < conv_width; ++lane) {
                                unit.offsets.push_back(ssmBytes()
                                                       + (step * qkv + channel_base + h * linear_dim + dim) * conv_width
                                                       + lane);
                            }
                        }
                    }
                    result.push_back(std::move(unit));
                }
                channel_base += count / tp_size * linear_dim;
            }
        }
        return result;
    }

    std::vector<uint8_t> payload(int layer, const std::string& tag, size_t logical) const {
        const auto           layout = units(tag);
        std::vector<uint8_t> bytes(tag == "full" ? 2 * fullHeads() * block_tokens * head_dim * 2 :
                                                   ssmBytes() + convBytes());
        std::vector<bool>    covered(bytes.size(), false);
        for (const auto& unit : layout) {
            const int component = unit.component == "K"      ? 1 :
                                  unit.component == "V"      ? 2 :
                                  unit.component == "SSM"    ? 3 :
                                  unit.component == "conv_Q" ? 4 :
                                  unit.component == "conv_K" ? 5 :
                                                               6;
            for (size_t i = 0; i < unit.offsets.size(); ++i) {
                // A unit's ordered index is token/dimension/lane for FULL,
                // value-dim/key-dim/lane for SSM, history/dim/lane for conv.
                uint64_t state = seed ^ (static_cast<uint64_t>(layer + 1) * UINT64_C(0xd6e8feb86659fd93));
                state ^= static_cast<uint64_t>(component) * UINT64_C(0xa0761d6478bd642f);
                state ^= static_cast<uint64_t>(unit.global_head + 1) * UINT64_C(0xe7037ed1a0b428db);
                state ^= (logical + 1) * UINT64_C(0x8ebc6af09c88c6e3);
                state ^= (i + 1) * UINT64_C(0x589965cc75374cc3);
                state = (state ^ (state >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                state = (state ^ (state >> 27)) * UINT64_C(0x94d049bb133111eb);
                state ^= state >> 31;
                require(unit.offsets[i] < bytes.size() && !covered[unit.offsets[i]],
                        "independent Next oracle overlapping or invalid byte coordinate");
                covered[unit.offsets[i]] = true;
                bytes[unit.offsets[i]]   = static_cast<uint8_t>(state);
            }
        }
        require(std::all_of(covered.begin(), covered.end(), [](bool value) { return value; }),
                "independent Next oracle has uncovered payload bytes");
        return bytes;
    }
};

inline CheckResult checkNext(KVCacheManager& manager, const KVCacheResource& resource, const NextOracle& oracle) {
    CheckResult result;
    for (int layer = 0; layer < resource.layerNum(); ++layer) {
        for (const auto& tag : resource.groupTagsForLayer(layer)) {
            const auto& blocks = resource.blocks(tag);
            for (size_t logical = 0; logical < blocks.size(); ++logical) {
                if (isNullBlockIdx(blocks[logical]))
                    continue;
                const auto expected = oracle.payload(layer, tag, logical);
                const auto actual =
                    readBytes(manager.convertIndexToAddr(layer, tag, blocks[logical]).kv_addr, expected.size());
                result.matches &= actual == expected;
                result.expected = signature(expected.data(), expected.size(), result.expected);
                result.observed = signature(actual.data(), actual.size(), result.observed);
                result.bytes += actual.size();
                const std::string identity = "\"layer\":" + std::to_string(layer) + ",\"group\":" + quote(tag)
                                             + ",\"logical_block\":" + std::to_string(logical);
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
                        std::cerr << "Next payload mismatch layer=" << layer << " group=" << tag
                                  << " logical_block=" << logical << " component=" << unit.component
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

inline bool nextGuards(KVCacheManager& manager, const KVCacheResource& resource, const NextOracle& oracle) {
    for (const auto& group : manager.cacheConfig().groups()) {
        const auto&         blocks = resource.blocks(group.tag);
        const std::set<int> live(blocks.begin(), blocks.end());
        for (int layer : manager.cacheConfig().layerIdsForGroup(group.tag)) {
            for (uint32_t block = 0; block < group.block_num; ++block) {
                if (live.count(block))
                    continue;
                const auto bytes = readBytes(manager.convertIndexToAddr(layer, group.tag, block).kv_addr,
                                             oracle.payload(layer, group.tag, 0).size());
                if (!std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0xa5; }))
                    return false;
            }
        }
    }
    return true;
}

inline std::pair<BatchKVCacheResourcePtr, CompleteTokenIdsPtr>
nextAllocate(KVCacheManager& manager, int tokens, int block_tokens, int64_t request) {
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
            "production Next malloc failed status=" + std::to_string(static_cast<int>(allocated.status)));
    return {resource, complete};
}

}  // namespace rtp_llm::cache_smoke
