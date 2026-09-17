#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

struct MlaHostCacheBudget {
    size_t hbm_bytes;
    size_t host_bytes;
    size_t mla_block_bytes;        // Sum over MLA layers; excludes indexer/KDA.
    size_t auxiliary_block_bytes;  // HBM bytes per logical block, including indexer.
    size_t mla_layers;
    size_t allocator_tokens;
    size_t kernel_tokens;
    size_t query_tokens;  // Maximum flattened decode/verify batch.
    size_t topk;          // Expanded raw tokens, 2051 for GLM53.
    size_t requested_resident_tokens = 0;
};

struct MlaHostCachePlan {
    uint32_t hbm_blocks;
    uint32_t host_blocks;
    size_t   resident_tokens;
    size_t   hbm_bytes;
};

// Pure sizing policy: fixed KDA/auxiliary pools are reserved by the caller.
// Each MLA layer has its own top-k map; all logical indexer pages stay in HBM.
inline MlaHostCachePlan planMlaHostCache(const MlaHostCacheBudget& b) {
    const auto mul = [](size_t a, size_t c) {
        if (c && a > std::numeric_limits<size_t>::max() / c) {
            throw std::invalid_argument("MLA host cache size overflow");
        }
        return a * c;
    };
    const auto add = [](size_t a, size_t c) {
        if (a > std::numeric_limits<size_t>::max() - c) {
            throw std::invalid_argument("MLA host cache size overflow");
        }
        return a + c;
    };
    constexpr size_t max_tokens = std::numeric_limits<int32_t>::max();
    if (!b.allocator_tokens || !b.kernel_tokens || b.allocator_tokens % b.kernel_tokens || !b.mla_block_bytes
        || b.mla_block_bytes % b.allocator_tokens || !b.mla_layers || !b.query_tokens || !b.topk) {
        throw std::invalid_argument("invalid MLA host cache geometry");
    }
    const size_t selected = mul(b.query_tokens, b.topk);
    const size_t resident = b.requested_resident_tokens ?
                                b.requested_resident_tokens :
                                mul(add(selected, b.kernel_tokens - 1) / b.kernel_tokens, b.kernel_tokens);
    if (resident < selected || resident % b.kernel_tokens || resident > max_tokens) {
        throw std::invalid_argument("MLA working set must cover decode/verify top-k and contain whole kernel pages");
    }
    const size_t row_bytes = b.mla_block_bytes / b.allocator_tokens;
    // Resident tags/version/protection + admission output/miss buffers. The
    // additional per-layer margin covers scalar state and allocator alignment.
    const size_t allocated_resident =
        mul(add(resident, b.allocator_tokens - 1) / b.allocator_tokens, b.allocator_tokens);
    const size_t working_bytes = add(mul(allocated_resident, row_bytes), mul(resident, mul(b.mla_layers, 32)));
    const size_t overhead      = add(mul(b.mla_layers, 4096), mul(b.query_tokens, row_bytes));
    const size_t fixed         = add(working_bytes, overhead);
    // Mapping + owner: two int32 per logical token per MLA layer. Reserve an
    // int64 generation snapshot even when the optional snapshot is disabled.
    const size_t logical_bytes =
        add(b.auxiliary_block_bytes, mul(b.mla_layers, add(mul(b.allocator_tokens, 8), sizeof(int64_t))));
    const size_t full_block_bytes = add(b.mla_block_bytes, logical_bytes);
    const size_t minimum_hbm      = mul(2, full_block_bytes);  // Reserved block 0 + one usable block.
    if (b.hbm_bytes <= fixed || b.hbm_bytes - fixed <= minimum_hbm) {
        throw std::invalid_argument("MLA HBM budget cannot hold the decode working set and complete blocks");
    }
    const size_t remaining   = b.hbm_bytes - fixed;
    const size_t host_blocks = std::min(b.host_bytes / b.mla_block_bytes, (remaining - minimum_hbm) / logical_bytes);
    const size_t hbm_blocks  = (remaining - mul(host_blocks, logical_bytes)) / full_block_bytes;
    if (!host_blocks || hbm_blocks < 2 || add(host_blocks, hbm_blocks) > max_tokens / b.allocator_tokens
        || hbm_blocks > (max_tokens - resident) / b.allocator_tokens) {
        throw std::invalid_argument("MLA host cache capacity is empty or exceeds int32 token IDs");
    }
    const size_t used = add(fixed, add(mul(hbm_blocks, full_block_bytes), mul(host_blocks, logical_bytes)));
    return {static_cast<uint32_t>(hbm_blocks), static_cast<uint32_t>(host_blocks), resident, used};
}

}  // namespace rtp_llm
