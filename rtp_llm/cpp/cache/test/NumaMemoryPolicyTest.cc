#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/NumaMemoryPolicy.h"

namespace rtp_llm {
namespace test {
namespace {

constexpr size_t kMiB = 1024 * 1024;

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
    ASSERT_TRUE(allowed.success) << allowed.error_message;
    if (allowed.allowed_nodes.size() < 2) {
        GTEST_SKIP() << "process is allowed to allocate memory on only one NUMA node";
    }

    ASSERT_EQ(::setenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register", 1), 0);
    ASSERT_EQ(::setenv("RTP_LLM_HOST_BLOCK_POOL_NUMA_POLICY", "interleave", 1), 0);

    auto      config = BlockPoolConfigHelper::createConfig(1, 16, 4 * kMiB, rtp_llm::TYPE_INT8);
    BlockPool pool(config, AllocationType::HOST);
    ASSERT_TRUE(pool.init());
    ASSERT_EQ(pool.where(), MemoryType::MEMORY_CPU_PINNED);

    expectPagesOnEveryAllowedNode(allowed, numaMappingInfo(pool.getBaseAddress()));

    ASSERT_EQ(::unsetenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE"), 0);
    ASSERT_EQ(::unsetenv("RTP_LLM_HOST_BLOCK_POOL_NUMA_POLICY"), 0);
}

TEST(NumaMemoryPolicyTest, InvalidHostBlockPoolNumaPolicyFailsFast) {
    ASSERT_EQ(::setenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register", 1), 0);
    ASSERT_EQ(::setenv("RTP_LLM_HOST_BLOCK_POOL_NUMA_POLICY", "invalid", 1), 0);

    auto      config = BlockPoolConfigHelper::createConfig(1, 2, 4 * kMiB, rtp_llm::TYPE_INT8);
    BlockPool pool(config, AllocationType::HOST);
    EXPECT_THROW(pool.init(), std::invalid_argument);

    ASSERT_EQ(::unsetenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE"), 0);
    ASSERT_EQ(::unsetenv("RTP_LLM_HOST_BLOCK_POOL_NUMA_POLICY"), 0);
}

}  // namespace test
}  // namespace rtp_llm
