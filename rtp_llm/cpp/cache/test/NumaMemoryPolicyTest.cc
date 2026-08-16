#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/NumaMemoryPolicy.h"

namespace rtp_llm {
namespace test {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        if (const char* current = std::getenv(name)) {
            original_value_ = current;
        }
        if (::setenv(name, value, 1) != 0) {
            throw std::runtime_error(std::string("setenv failed for ") + name + ": " + std::strerror(errno));
        }
    }

    ~ScopedEnvVar() {
        if (original_value_.has_value()) {
            (void)::setenv(name_.c_str(), original_value_->c_str(), 1);
        } else {
            (void)::unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&)            = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    std::string                name_;
    std::optional<std::string> original_value_;
};

bool numaPolicySyscallIsBlocked(const NumaInterleavePolicyResult& result) {
    return !result.success && (result.error_number == EPERM || result.error_number == ENOSYS);
}

struct NumaMappingInfo {
    std::string           line;
    std::map<int, size_t> pages_by_node;
};

uintptr_t mappingStart(const void* address) {
    std::ifstream maps("/proc/self/maps");
    std::string   line;
    const auto    target = reinterpret_cast<uintptr_t>(address);
    while (std::getline(maps, line)) {
        uintptr_t begin = 0;
        uintptr_t end   = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2 && target >= begin && target < end) {
            return begin;
        }
    }
    return 0;
}

NumaMappingInfo numaMappingInfo(const void* address) {
    const uintptr_t begin = mappingStart(address);
    if (begin == 0) {
        return {};
    }

    std::ifstream numa_maps("/proc/self/numa_maps");
    std::string   line;
    while (std::getline(numa_maps, line)) {
        uintptr_t line_begin = 0;
        if (std::sscanf(line.c_str(), "%lx", &line_begin) != 1 || line_begin != begin) {
            continue;
        }

        NumaMappingInfo info;
        info.line = line;
        std::istringstream tokens(line);
        std::string        token;
        while (tokens >> token) {
            int    node  = -1;
            size_t pages = 0;
            if (std::sscanf(token.c_str(), "N%d=%zu", &node, &pages) == 2) {
                info.pages_by_node[node] = pages;
            }
        }
        return info;
    }
    return {};
}

void expectPagesOnEveryAllowedNode(const NumaInterleavePolicyResult& result, const NumaMappingInfo& info) {
    ASSERT_FALSE(info.line.empty());
    EXPECT_NE(info.line.find("interleave"), std::string::npos) << info.line;
    for (int node : result.allowed_nodes) {
        const auto it = info.pages_by_node.find(node);
        ASSERT_NE(it, info.pages_by_node.end()) << "node=" << node << " numa_maps='" << info.line << "'";
        EXPECT_GT(it->second, 0u) << "node=" << node << " numa_maps='" << info.line << "'";
    }
}

}  // namespace

TEST(NumaMemoryPolicyTest, AnonymousMappingIsInterleavedAcrossAllowedNodes) {
    constexpr size_t size = 64 * kMiB;
    void*            ptr  = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(ptr, MAP_FAILED) << std::strerror(errno);

    const auto result = applyAllowedNumaInterleavePolicy(ptr, size);
    if (!result.success) {
        (void)::munmap(ptr, size);
        if (numaPolicySyscallIsBlocked(result)) {
            GTEST_SKIP() << result.error_message;
        }
        FAIL() << result.error_message;
    }
    if (result.allowed_nodes.size() < 2) {
        (void)::munmap(ptr, size);
        GTEST_SKIP() << "process is allowed to allocate memory on only one NUMA node";
    }
    ASSERT_TRUE(result.applied);

    const long page_size = ::sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    auto* bytes = static_cast<volatile uint8_t*>(ptr);
    for (size_t offset = 0; offset < size; offset += static_cast<size_t>(page_size)) {
        bytes[offset] = static_cast<uint8_t>(offset / static_cast<size_t>(page_size));
    }

    expectPagesOnEveryAllowedNode(result, numaMappingInfo(ptr));
    ASSERT_EQ(::munmap(ptr, size), 0);
}

TEST(NumaMemoryPolicyTest, RegisteredHostBlockPoolKeepsInterleavePolicy) {
    constexpr size_t size  = 64 * kMiB;
    void*            probe = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(probe, MAP_FAILED) << std::strerror(errno);
    const auto allowed = applyAllowedNumaInterleavePolicy(probe, size);
    ASSERT_EQ(::munmap(probe, size), 0);
    if (numaPolicySyscallIsBlocked(allowed)) {
        GTEST_SKIP() << allowed.error_message;
    }
    ASSERT_TRUE(allowed.success) << allowed.error_message;
    if (allowed.allowed_nodes.size() < 2) {
        GTEST_SKIP() << "process is allowed to allocate memory on only one NUMA node";
    }

    ScopedEnvVar pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register");
    ScopedEnvVar interleave("RTP_LLM_HOST_BLOCK_POOL_INTERLEAVE", "1");

    uintptr_t pool_address = 0;
    {
        auto      config = BlockPoolConfigHelper::createConfig(1, 16, 4 * kMiB, rtp_llm::TYPE_INT8);
        BlockPool pool(config, AllocationType::HOST);
        ASSERT_TRUE(pool.init());
        ASSERT_EQ(pool.where(), MemoryType::MEMORY_CPU_PINNED);

#if USING_CUDA
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, pool.getBaseAddress()), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
#endif
        std::memset(pool.getBaseAddress(), 0, size);
        expectPagesOnEveryAllowedNode(allowed, numaMappingInfo(pool.getBaseAddress()));
        pool_address = reinterpret_cast<uintptr_t>(pool.getBaseAddress());
    }
    EXPECT_EQ(mappingStart(reinterpret_cast<void*>(pool_address)), 0u);
}

TEST(NumaMemoryPolicyTest, RegisteredHostBlockPoolSupportsDisabledInterleave) {
    ScopedEnvVar pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register");
    ScopedEnvVar interleave("RTP_LLM_HOST_BLOCK_POOL_INTERLEAVE", "0");

    uintptr_t pool_address = 0;
    {
        auto      config = BlockPoolConfigHelper::createConfig(1, 4, 4 * kMiB, rtp_llm::TYPE_INT8);
        BlockPool pool(config, AllocationType::HOST);
        ASSERT_TRUE(pool.init());
        ASSERT_EQ(pool.where(), MemoryType::MEMORY_CPU_PINNED);
#if USING_CUDA
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, pool.getBaseAddress()), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
#endif
        pool_address = reinterpret_cast<uintptr_t>(pool.getBaseAddress());
    }
    EXPECT_EQ(mappingStart(reinterpret_cast<void*>(pool_address)), 0u);
}

TEST(NumaMemoryPolicyTest, StrictInterleaveFailsWhenNumaPolicySyscallsAreBlocked) {
    constexpr size_t probe_size = 4 * kMiB;
    void*            probe = ::mmap(nullptr, probe_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(probe, MAP_FAILED) << std::strerror(errno);
    const auto policy_result = applyAllowedNumaInterleavePolicy(probe, probe_size);
    ASSERT_EQ(::munmap(probe, probe_size), 0);
    if (!numaPolicySyscallIsBlocked(policy_result)) {
        GTEST_SKIP() << "NUMA policy syscalls are available in this environment";
    }

    ScopedEnvVar pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register");
    ScopedEnvVar interleave("RTP_LLM_HOST_BLOCK_POOL_INTERLEAVE", "1");

    auto      config = BlockPoolConfigHelper::createConfig(1, 4, 4 * kMiB, rtp_llm::TYPE_INT8);
    BlockPool pool(config, AllocationType::HOST);
    EXPECT_THROW(pool.init(), std::runtime_error);
}

TEST(NumaMemoryPolicyTest, InvalidMappingReturnsStructuredError) {
    const auto result = applyAllowedNumaInterleavePolicy(nullptr, 0);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.applied);
    EXPECT_EQ(result.error_number, EINVAL);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(NumaMemoryPolicyTest, InvalidHostBlockPoolInterleaveFlagFailsFast) {
    ScopedEnvVar pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register");
    ScopedEnvVar interleave("RTP_LLM_HOST_BLOCK_POOL_INTERLEAVE", "invalid");

    auto      config = BlockPoolConfigHelper::createConfig(1, 2, 4 * kMiB, rtp_llm::TYPE_INT8);
    BlockPool pool(config, AllocationType::HOST);
    EXPECT_THROW(pool.init(), std::invalid_argument);
}

TEST(NumaMemoryPolicyTest, ExplicitInterleaveRejectsPinnedAllocatorMode) {
    ScopedEnvVar pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "allocator");
    ScopedEnvVar interleave("RTP_LLM_HOST_BLOCK_POOL_INTERLEAVE", "1");

    auto      config = BlockPoolConfigHelper::createConfig(1, 2, 4 * kMiB, rtp_llm::TYPE_INT8);
    BlockPool pool(config, AllocationType::HOST);
    EXPECT_THROW(pool.init(), std::invalid_argument);
}

}  // namespace test
}  // namespace rtp_llm
