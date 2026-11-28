// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {
// Bounds the unretired sequence range, including out-of-order completions.
inline constexpr uint64_t kHixlTransferWindow = 1024;

struct HixlMemoryRegion {
    uintptr_t address{0};
    uint64_t  bytes{0};
    uint64_t  id{0};
};

// One descriptor per cache block, validated against registered regions.
struct HixlTransferSlice {
    uintptr_t local_address;
    uintptr_t remote_address;
    size_t    bytes;
};

}  // namespace rtp_llm
