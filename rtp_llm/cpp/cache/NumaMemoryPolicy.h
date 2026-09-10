#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rtp_llm {

struct NumaInterleavePolicyResult {
    bool             success = false;
    bool             applied = false;
    std::vector<int> allowed_nodes;
    int              error_number = 0;
    std::string      error_message;
};

// Applies MPOL_INTERLEAVE to an existing, not-yet-populated mapping using all
// memory nodes allowed by the process' current cpuset/cgroup.
//
// success=true, applied=false means the process can use only one NUMA node, so
// interleaving is unnecessary. The caller must apply this before first touch.
NumaInterleavePolicyResult applyAllowedNumaInterleavePolicy(void* address, size_t size);

}  // namespace rtp_llm
