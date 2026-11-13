#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"

namespace rtp_llm::cache_smoke {

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline std::string smokeBackend() {
    const char*       selected = std::getenv("CACHE_SMOKE_BACKEND");
    const std::string backend  = selected ? selected : "tcp";
    require(backend == "tcp" || backend == "barex_rdma", "invalid Cache smoke backend");
    return backend;
}

inline std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\' << ch;
        } else if (ch < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
        } else {
            out << ch;
        }
    }
    out << '"';
    return out.str();
}

inline std::string hex(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

inline uint64_t signature(const uint8_t* bytes, size_t size, uint64_t hash = UINT64_C(14695981039346656037)) {
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

inline std::vector<uint8_t> readBytes(void* address, size_t size) {
    std::vector<uint8_t> bytes(size);
    require(cudaMemcpy(bytes.data(), address, size, cudaMemcpyDeviceToHost) == cudaSuccess, "DEVICE read failed");
    return bytes;
}

inline void writeBytes(void* address, const std::vector<uint8_t>& bytes) {
    require(cudaMemcpy(address, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess,
            "DEVICE write failed");
}

// Thin test-only adapter: every transport partition and address is resolved by
// the real manager. The checker below never calls this partition converter.
class ManagerConverter final: public LayerBlockConverter {
public:
    explicit ManagerConverter(std::shared_ptr<KVCacheManager> manager): manager_(std::move(manager)) {}

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer, const std::string& tag, int block, int count, int partition) const override {
        auto result = manager_->convertIndexToBuffer(layer, tag, block, count, partition);
        result.erase(std::remove_if(result.begin(),
                                    result.end(),
                                    [](const BlockInfo& info) { return info.addr == nullptr || info.size_bytes == 0; }),
                     result.end());
        return result;
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        std::vector<std::pair<BlockInfo, size_t>> result;
        std::set<void*>                           seen;
        const auto                                layout = manager_->allLayerCacheBase();
        for (const auto& [tag, group] : layout.groups()) {
            (void)tag;
            for (const auto& layer : group.layers()) {
                for (const auto& tensor : {layer.kv_addr, layer.kv_scale_addr}) {
                    if (!tensor.defined() || tensor.nbytes() == 0 || !seen.insert(tensor.data_ptr()).second) {
                        continue;
                    }
                    BlockInfo info;
                    info.addr         = tensor.data_ptr();
                    info.size_bytes   = tensor.nbytes();
                    info.is_cuda      = tensor.is_cuda();
                    info.device_index = tensor.is_cuda() ? tensor.get_device() : 0;
                    info.scalar_type  = static_cast<int32_t>(tensor.scalar_type());
                    result.emplace_back(info, info.size_bytes);
                }
            }
        }
        return result;
    }

private:
    std::shared_ptr<KVCacheManager> manager_;
};

// Actual manager tensors, deduplicated by the production-backed converter.
inline std::string allocatedStorageJson(const LayerBlockConverter& converter) {
    size_t device_bytes = 0, host_bytes = 0, device_buffers = 0, host_buffers = 0;
    for (const auto& [info, size] : converter.getAllBuffers()) {
        (info.is_cuda ? device_bytes : host_bytes) += size;
        (info.is_cuda ? device_buffers : host_buffers) += 1;
    }
    return "{\"device_bytes\":" + std::to_string(device_bytes) + ",\"host_bytes\":" + std::to_string(host_bytes)
           + ",\"device_buffers\":" + std::to_string(device_buffers)
           + ",\"host_buffers\":" + std::to_string(host_buffers) + "}";
}

inline size_t checkedPayloadSize(std::initializer_list<size_t> factors) {
    size_t bytes = 1;
    for (const size_t factor : factors) {
        require(factor > 0 && bytes <= std::numeric_limits<size_t>::max() / factor,
                "invalid or overflowing smoke payload size");
        bytes *= factor;
    }
    return bytes;
}

struct PayloadOracle {
    uint64_t seed;
    int      heads;
    int      head_dim;
    int      block_tokens;
    size_t   element_bytes;
    bool     scales;
    int      first_global_head = 0;

    bool mla        = false;
    int  latent_dim = 0;
    int  rope_dim   = 0;

    size_t mlaStride() const {
        require(latent_dim > 0 && rope_dim > 0, "invalid MLA payload dimensions");
        return element_bytes == 1 ?
                   size_t(latent_dim) + size_t(latent_dim / 128) * sizeof(float) + size_t(rope_dim) * 2 :
                   checkedPayloadSize({size_t(latent_dim) + size_t(rope_dim), element_bytes});
    }

    std::vector<uint8_t> mlaPayload(int layer, int logical_block) const {
        require(block_tokens > 0 && logical_block >= 0, "invalid MLA payload coordinates");
        std::vector<uint8_t> result(checkedPayloadSize({size_t(block_tokens), mlaStride()}));
        for (int token = 0; token < block_tokens; ++token) {
            for (size_t byte = 0; byte < mlaStride(); ++byte) {
                uint64_t state = seed ^ (static_cast<uint64_t>(layer + 1) * UINT64_C(0xd6e8feb86659fd93));
                state ^= (uint64_t(logical_block) * uint64_t(block_tokens) + uint64_t(token) + 1)
                         * UINT64_C(0x8ebc6af09c88c6e3);
                state ^= (byte + 1) * UINT64_C(0x589965cc75374cc3);
                state = (state ^ (state >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                state = (state ^ (state >> 27)) * UINT64_C(0x94d049bb133111eb);
                state ^= state >> 31;
                result[token * mlaStride() + byte] = static_cast<uint8_t>(state);
            }
            if (element_bytes == 1) {
                // FP8 MLA stores FP32 quantization scales inline, then BF16
                // RoPE. Both are full payload, never separate MHA scales.
                for (int group = 0; group < latent_dim / 128; ++group) {
                    const float scale = 0.25f + static_cast<float>(result[token * mlaStride() + group]) / 256.0f;
                    std::memcpy(result.data() + token * mlaStride() + latent_dim + group * sizeof(float),
                                &scale,
                                sizeof(scale));
                }
            }
        }
        return result;
    }

    // Head-major storage is the production MHA kernel contract. Generate each
    // byte from layer/component/global head/global token/dimension/byte-lane.
    // SplitMix64 expands logical identity, never rank or physical block ID.
    std::vector<uint8_t> payload(int layer, int logical_block, bool scale) const {
        if (mla) {
            require(!scale, "MLA has no separate MHA K/V scale tensor");
            return mlaPayload(layer, logical_block);
        }
        require(heads > 0 && block_tokens > 0 && head_dim > 0 && logical_block >= 0, "invalid MHA payload dimensions");
        const size_t         dim   = scale ? 1 : static_cast<size_t>(head_dim);
        const size_t         width = scale ? sizeof(float) : element_bytes;
        std::vector<uint8_t> result(checkedPayloadSize({2, size_t(heads), size_t(block_tokens), dim, width}));
        size_t               offset = 0;
        for (int component = 0; component < 2; ++component) {
            for (int head = 0; head < heads; ++head) {
                for (int token = 0; token < block_tokens; ++token) {
                    for (size_t d = 0; d < dim; ++d) {
                        uint64_t state = seed ^ (static_cast<uint64_t>(layer + 1) * UINT64_C(0xd6e8feb86659fd93));
                        state ^= static_cast<uint64_t>(component + 1 + (scale ? 2 : 0)) * UINT64_C(0xa0761d6478bd642f);
                        state ^= static_cast<uint64_t>(first_global_head + head + 1) * UINT64_C(0xe7037ed1a0b428db);
                        state ^= (uint64_t(logical_block) * uint64_t(block_tokens) + uint64_t(token) + 1)
                                 * UINT64_C(0x8ebc6af09c88c6e3);
                        state ^= (d + 1) * UINT64_C(0x589965cc75374cc3);
                        state = (state ^ (state >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                        state = (state ^ (state >> 27)) * UINT64_C(0x94d049bb133111eb);
                        state ^= state >> 31;
                        for (size_t lane = 0; lane < width; ++lane) {
                            result[offset++] = static_cast<uint8_t>(state >> (lane * 8));
                        }
                    }
                }
            }
        }
        return result;
    }
};

struct CheckResult {
    bool        matches  = true;
    uint64_t    expected = UINT64_C(14695981039346656037);
    uint64_t    observed = UINT64_C(14695981039346656037);
    size_t      bytes    = 0;
    std::string detail   = "[";
    std::string units    = "[";
};

inline CheckResult
checkMlaPayload(KVCacheManager& manager, const KVCacheResource& resource, const PayloadOracle& oracle) {
    CheckResult                                       result;
    const auto&                                       blocks       = resource.blocks("default");
    const bool                                        fp8          = oracle.element_bytes == 1;
    const size_t                                      latent_bytes = oracle.latent_dim * oracle.element_bytes;
    const size_t                                      scale_bytes = fp8 ? (oracle.latent_dim / 128) * sizeof(float) : 0;
    const size_t                                      rope_bytes  = oracle.rope_dim * (fp8 ? 2 : oracle.element_bytes);
    const std::vector<std::pair<std::string, size_t>> components =
        fp8 ? std::vector<std::pair<std::string, size_t>>{{"latent", latent_bytes},
                                                          {"inline_scale", scale_bytes},
                                                          {"RoPE", rope_bytes}} :
              std::vector<std::pair<std::string, size_t>>{{"latent", latent_bytes}, {"RoPE", rope_bytes}};
    for (uint32_t layer = 0; layer < manager.cacheConfig().layer_num; ++layer) {
        for (size_t logical = 0; logical < blocks.size(); ++logical) {
            const auto expected = oracle.payload(layer, logical, false);
            const auto address  = manager.convertIndexToAddr(layer, "default", blocks[logical]);
            const auto actual   = readBytes(address.kv_addr, expected.size());
            result.matches &= expected == actual;
            result.expected = signature(expected.data(), expected.size(), result.expected);
            result.observed = signature(actual.data(), actual.size(), result.observed);
            result.bytes += actual.size();
            if (result.detail.size() > 1)
                result.detail += ',';
            result.detail += "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                             + std::to_string(logical) + ",\"physical_block\":" + std::to_string(blocks[logical])
                             + ",\"component\":\"MLA_packed\",\"bytes\":" + std::to_string(actual.size())
                             + ",\"expected\":" + quote(hex(signature(expected.data(), expected.size())))
                             + ",\"observed\":" + quote(hex(signature(actual.data(), actual.size())))
                             + ",\"matches\":" + (actual == expected ? "true" : "false") + "}";
            size_t component_offset = 0;
            for (const auto& [component, width] : components) {
                std::vector<uint8_t> expected_component, actual_component;
                for (int token = 0; token < oracle.block_tokens; ++token) {
                    const size_t offset = token * oracle.mlaStride() + component_offset;
                    expected_component.insert(
                        expected_component.end(), expected.begin() + offset, expected.begin() + offset + width);
                    actual_component.insert(
                        actual_component.end(), actual.begin() + offset, actual.begin() + offset + width);
                }
                if (result.units.size() > 1)
                    result.units += ',';
                result.units +=
                    "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                    + std::to_string(logical) + ",\"global_head\":0,\"component\":" + quote(component)
                    + ",\"scale\":" + (component == "inline_scale" ? "true" : "false")
                    + ",\"bytes\":" + std::to_string(actual_component.size())
                    + ",\"expected\":" + quote(hex(signature(expected_component.data(), expected_component.size())))
                    + ",\"observed\":" + quote(hex(signature(actual_component.data(), actual_component.size())))
                    + ",\"matches\":" + (actual_component == expected_component ? "true" : "false") + "}";
                component_offset += width;
            }
            if (actual != expected) {
                const size_t byte =
                    std::mismatch(actual.begin(), actual.end(), expected.begin()).first - actual.begin();
                const size_t within = byte % oracle.mlaStride();
                std::cerr << "MLA payload mismatch layer=" << layer << " group=default logical_block=" << logical
                          << " global_token=" << logical * oracle.block_tokens + byte / oracle.mlaStride()
                          << " component="
                          << (within < latent_bytes               ? "latent" :
                              within < latent_bytes + scale_bytes ? "inline_scale" :
                                                                    "RoPE")
                          << " packed_byte=" << within << '\n';
            }
        }
    }
    result.detail += ']';
    result.units += ']';
    return result;
}

inline CheckResult checkPayload(KVCacheManager& manager, const KVCacheResource& resource, const PayloadOracle& oracle) {
    if (oracle.mla)
        return checkMlaPayload(manager, resource, oracle);
    CheckResult result;
    bool        first  = true;
    const auto& blocks = resource.blocks("default");
    for (uint32_t layer = 0; layer < manager.cacheConfig().layer_num; ++layer) {
        for (size_t logical = 0; logical < blocks.size(); ++logical) {
            auto address = manager.convertIndexToAddr(layer, "default", blocks[logical]);
            for (bool scale : {false, true}) {
                if (scale && !oracle.scales) {
                    continue;
                }
                auto expected = oracle.payload(layer, logical, scale);
                auto actual   = readBytes(scale ? address.kv_scale_addr : address.kv_addr, expected.size());
                result.matches &= expected == actual;
                result.expected = signature(expected.data(), expected.size(), result.expected);
                result.observed = signature(actual.data(), actual.size(), result.observed);
                result.bytes += actual.size();
                if (!first) {
                    result.detail += ',';
                }
                first = false;
                result.detail += "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                                 + std::to_string(logical) + ",\"physical_block\":" + std::to_string(blocks[logical])
                                 + ",\"first_global_head\":" + std::to_string(oracle.first_global_head)
                                 + ",\"head_count\":" + std::to_string(oracle.heads) + ",\"component\":"
                                 + quote(scale ? "K/V_scale" : "K/V") + ",\"bytes\":" + std::to_string(actual.size())
                                 + ",\"expected\":" + quote(hex(signature(expected.data(), expected.size())))
                                 + ",\"observed\":" + quote(hex(signature(actual.data(), actual.size())))
                                 + ",\"matches\":" + (actual == expected ? "true" : "false") + "}";
                // Cross-TP comparison uses individual global heads and K/V
                // components, whose bytes are invariant under partitioning.
                const size_t head_bytes = actual.size() / (2 * oracle.heads);
                for (int component = 0; component < 2; ++component) {
                    for (int head = 0; head < oracle.heads; ++head) {
                        const size_t offset = (component * oracle.heads + head) * head_bytes;
                        if (result.units.size() > 1)
                            result.units += ',';
                        result.units += "{\"layer\":" + std::to_string(layer)
                                        + ",\"group\":\"default\",\"logical_block\":" + std::to_string(logical)
                                        + ",\"global_head\":" + std::to_string(oracle.first_global_head + head)
                                        + ",\"component\":" + quote(component == 0 ? "K" : "V") + ",\"scale\":"
                                        + (scale ? "true" : "false") + ",\"bytes\":" + std::to_string(head_bytes)
                                        + ",\"expected\":" + quote(hex(signature(expected.data() + offset, head_bytes)))
                                        + ",\"observed\":" + quote(hex(signature(actual.data() + offset, head_bytes)))
                                        + ",\"matches\":"
                                        + (std::equal(expected.begin() + offset,
                                                      expected.begin() + offset + head_bytes,
                                                      actual.begin() + offset) ?
                                               "true" :
                                               "false")
                                        + "}";
                    }
                }
                if (actual != expected) {
                    const auto   mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
                    const size_t byte     = mismatch.first - actual.begin();
                    const size_t width    = scale ? sizeof(float) : oracle.element_bytes;
                    const size_t dim      = scale ? 1 : static_cast<size_t>(oracle.head_dim);
                    const size_t half     = actual.size() / 2;
                    const size_t element  = (byte % half) / width;
                    std::cerr << "payload mismatch layer=" << layer << " group=default logical_block=" << logical
                              << " physical_block=" << blocks[logical] << " scale=" << scale
                              << " component=" << (byte < half ? "K" : "V")
                              << " global_head=" << oracle.first_global_head + element / (dim * oracle.block_tokens)
                              << " global_token="
                              << logical * oracle.block_tokens + (element / dim) % oracle.block_tokens
                              << " dim=" << element % dim << " byte_lane=" << byte % width << '\n';
                }
            }
        }
    }
    result.detail += ']';
    result.units += ']';
    return result;
}

}  // namespace rtp_llm::cache_smoke
