#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/BlockIndex.h"
#include "rtp_llm/cpp/cache/CacheTier.h"

namespace rtp_llm {

struct TreeNode;

enum class ExternalMatchMode { BORROWED_SLOTS };
enum class ExternalBindingKind { READY = 0, JOIN = 1, LOAD = 2 };

struct ExternalJoinTarget {
    BlockIdxType slot;
    uint64_t     generation;
    uint64_t     operation;
};

using ExternalJoinLookup = std::function<std::optional<ExternalJoinTarget>(const TreeNode& node, size_t group)>;

struct ExternalMatchPosition {
    size_t group;
    size_t row;
};

struct ExternalMatchEndpoint {
    size_t                             tokens;
    std::vector<ExternalMatchPosition> needs;
};

struct ExternalPageGroup {
    size_t block_size;
    bool   last_checkpoint;  // A recurrent group only needs the last state.
};

struct ExternalMatchBinding {
    size_t              group;
    size_t              row;
    Tier                tier;
    BlockIdxType        slot;
    ExternalBindingKind kind;
    uint64_t            generation{0};
    uint64_t            operation{0};
};

struct ExternalBlockTreeMatchResult {
    size_t                            device_tokens{0};
    size_t                            total_tokens{0};
    std::vector<ExternalMatchBinding> device_bindings;
    std::vector<ExternalMatchBinding> bindings;
};

}  // namespace rtp_llm
