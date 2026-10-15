#include "rtp_llm/cpp/cache/NumaMemoryPolicy.h"

#include <cerrno>
#include <cstdint>
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

NumaInterleavePolicyResult bindAndVerifyNumaNode(void* address, size_t size, int node) {
    if (address == nullptr || size == 0 || node < 0 || static_cast<unsigned long>(node) >= kMaximumMaxNode) {
        return failureResult("bindAndVerifyNumaNode invalid arguments", EINVAL);
    }
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || reinterpret_cast<uintptr_t>(address) % static_cast<size_t>(page_size) != 0) {
        return failureResult("bindAndVerifyNumaNode invalid page alignment", EINVAL);
    }

    unsigned long              max_node = kInitialMaxNode;
    std::vector<unsigned long> allowed_mask;
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
    if (static_cast<unsigned long>(node) >= max_node
        || (allowed_mask[node / kBitsPerWord] & (1UL << (node % kBitsPerWord))) == 0) {
        return failureResult("GPU NUMA node is not allowed by the process cpuset", EACCES);
    }

    std::vector<unsigned long> target_mask(wordCount(max_node), 0);
    target_mask[node / kBitsPerWord] = 1UL << (node % kBitsPerWord);
    if (::syscall(SYS_mbind, address, size, MPOL_BIND, target_mask.data(), max_node, 0) != 0) {
        return failureResult("mbind(MPOL_BIND)", errno);
    }

    // memfd pages must be materialized before CUDA pins them and before PACE
    // advertises the NUMA hint. A successful mbind alone does not prove placement.
    auto*        bytes  = static_cast<volatile unsigned char*>(address);
    const size_t stride = static_cast<size_t>(page_size);
    for (size_t offset = 0; offset < size; offset += stride) {
        bytes[offset] = 0;
    }

    constexpr size_t kBatchPages = 1024;
    void*            pages[kBatchPages];
    int              status[kBatchPages];
    for (size_t offset = 0; offset < size;) {
        size_t count = 0;
        for (; count < kBatchPages && offset < size; ++count, offset += stride) {
            pages[count] = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) + offset);
        }
        if (::syscall(SYS_move_pages, 0, count, pages, nullptr, status, 0) != 0) {
            return failureResult("move_pages query", errno);
        }
        for (size_t i = 0; i < count; ++i) {
            if (status[i] != node) {
                NumaInterleavePolicyResult result =
                    failureResult("Host page is outside GPU NUMA node", status[i] < 0 ? -status[i] : EXDEV);
                result.error_message += ": expected=" + std::to_string(node) + " actual=" + std::to_string(status[i]);
                return result;
            }
        }
    }

    NumaInterleavePolicyResult result;
    result.success       = true;
    result.applied       = true;
    result.allowed_nodes = {node};
    return result;
}

}  // namespace rtp_llm
