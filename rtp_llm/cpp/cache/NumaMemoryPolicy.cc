#include "rtp_llm/cpp/cache/NumaMemoryPolicy.h"

#include <cerrno>
#include <climits>
#include <cstring>
#include <sstream>
#include <vector>

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

constexpr unsigned long kBitsPerWord    = sizeof(unsigned long) * CHAR_BIT;
constexpr unsigned long kInitialMaxNode = 64;
constexpr unsigned long kMaximumMaxNode = 4096;

size_t wordCount(unsigned long max_node) {
    return static_cast<size_t>((max_node + kBitsPerWord - 1) / kBitsPerWord);
}

std::vector<int> nodesFromMask(const std::vector<unsigned long>& mask, unsigned long max_node) {
    std::vector<int> nodes;
    for (unsigned long node = 0; node < max_node; ++node) {
        if ((mask[node / kBitsPerWord] & (1UL << (node % kBitsPerWord))) != 0) {
            nodes.push_back(static_cast<int>(node));
        }
    }
    return nodes;
}

std::string nodeList(const std::vector<int>& nodes) {
    std::ostringstream oss;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << nodes[i];
    }
    return oss.str();
}

NumaInterleavePolicyResult failureResult(const char* operation, int error_number) {
    NumaInterleavePolicyResult result;
    result.error_number  = error_number;
    result.error_message = std::string(operation) + " failed: errno=" + std::to_string(error_number) + " ("
                           + std::strerror(error_number) + ")";
    return result;
}

}  // namespace

NumaInterleavePolicyResult applyAllowedNumaInterleavePolicy(void* address, size_t size) {
    if (address == nullptr || size == 0) {
        return failureResult("applyAllowedNumaInterleavePolicy invalid mapping", EINVAL);
    }

    std::vector<unsigned long> allowed_mask;
    unsigned long              max_node = kInitialMaxNode;
    for (; max_node <= kMaximumMaxNode; max_node *= 2) {
        allowed_mask.assign(wordCount(max_node), 0);
        int mode = MPOL_DEFAULT;
        errno    = 0;
        if (::syscall(SYS_get_mempolicy, &mode, allowed_mask.data(), max_node, nullptr, MPOL_F_MEMS_ALLOWED) == 0) {
            break;
        }
        if (errno != EINVAL || max_node == kMaximumMaxNode) {
            return failureResult("get_mempolicy(MPOL_F_MEMS_ALLOWED)", errno);
        }
    }

    NumaInterleavePolicyResult result;
    result.allowed_nodes = nodesFromMask(allowed_mask, max_node);
    if (result.allowed_nodes.empty()) {
        return failureResult("get_mempolicy returned an empty allowed-node mask", ENODEV);
    }
    if (result.allowed_nodes.size() == 1) {
        result.success = true;
        return result;
    }

    errno = 0;
    if (::syscall(SYS_mbind, address, size, MPOL_INTERLEAVE, allowed_mask.data(), max_node, 0) != 0) {
        result.error_number  = errno;
        result.error_message = "mbind(MPOL_INTERLEAVE, nodes=" + nodeList(result.allowed_nodes)
                               + ") failed: errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")";
        return result;
    }

    result.success = true;
    result.applied = true;
    return result;
}

}  // namespace rtp_llm
