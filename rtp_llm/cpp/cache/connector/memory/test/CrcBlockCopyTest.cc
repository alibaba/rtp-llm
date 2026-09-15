#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace rtp_llm::test {
namespace {

uint32_t crc32c(const uint8_t* data, size_t bytes) {
    uint32_t crc = ~0u;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((0u - (crc & 1u)) & 0x82f63b78u);
        }
    }
    return ~crc;
}

// Test-only orchestration; a failed CPU decision must not reach gather/store or scatter.
CrcBlockCopyResult copyBlock(CrcBlockCopy&                        workspace,
                             void*                                host,
                             size_t                               bytes,
                             const std::vector<CrcBlockCopyTile>& tiles,
                             bool                                 to_device,
                             int64_t                              request_id          = 0,
                             const void*                          inherited           = nullptr,
                             const std::function<bool()>&         capture_failure     = {},
                             bool                                 host_is_pinned      = true,
                             bool                                 inherited_is_pinned = true) {
    if (to_device || inherited) {
        auto result = workspace.loadAndValidate(
            to_device ? host : inherited, bytes, to_device ? host_is_pinned : inherited_is_pinned, capture_failure);
        if (!result.success)
            return result;
        if (to_device) {
            workspace.scatter(bytes, tiles);
            return result;
        }
    }
    workspace.gather(bytes, tiles, inherited != nullptr);
    return workspace.store(host, bytes, request_id, host_is_pinned, capture_failure);
}

struct Fixture {
    size_t                        bytes{0};
    bool                          pinned{true};
    uint8_t *                     host{nullptr}, *device{nullptr};
    std::vector<CrcBlockCopyTile> tiles;
    std::unique_ptr<CrcBlockCopy> copy;
    int64_t                       request_id{23};

    explicit Fixture(const std::vector<size_t>& sizes,
                     size_t                     gap                = 0,
                     bool                       allocate_workspace = true,
                     bool                       pin_host           = true):
        pinned(pin_host) {
        for (auto n : sizes)
            bytes += n + gap;
        if (pinned) {
            if (cudaMallocHost(&host, CrcBlockCopy::storageBytes(bytes)) != cudaSuccess) {
                throw std::runtime_error("test pinned allocation failed");
            }
        } else {
            host = static_cast<uint8_t*>(std::malloc(CrcBlockCopy::storageBytes(bytes)));
        }
        if (!host || cudaMalloc(&device, bytes) != cudaSuccess) {
            throw std::runtime_error("test allocation failed");
        }
        size_t offset = 0;
        for (auto n : sizes) {
            tiles.push_back({device + offset, offset, n});
            offset += n + gap;
        }
        for (size_t i = 0; i < bytes; ++i)
            host[i] = (131 * i + 17 * (i >> 8) + 29) & 255;
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(nullptr);
        if (allocate_workspace) {
            copy = std::make_unique<CrcBlockCopy>(std::vector<size_t>{bytes}, tiles.size());
        }
    }
    ~Fixture() {
        copy.reset();
        if (pinned)
            cudaFreeHost(host);
        else
            std::free(host);
        cudaFree(device);
    }
    CrcBlockFooter footer() const {
        CrcBlockFooter result;
        std::memcpy(&result, host + CrcBlockCopy::footerOffset(bytes), sizeof(result));
        return result;
    }
    int64_t writerRequestId() const {
        int64_t result;
        std::memcpy(&result, host + CrcBlockCopy::transferBytes(bytes), sizeof(result));
        return result;
    }
    std::vector<uint8_t> deviceBytes() const {
        std::vector<uint8_t> result(bytes);
        if (cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("test readback failed");
        return result;
    }
};

TEST(CrcBlockCopyTest, StandardCrcAndRaggedRoundtrip) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    ASSERT_EQ(crc32c(reinterpret_cast<const uint8_t*>("123456789"), 9), 0xe3069283u);
    for (const auto& sizes :
         std::vector<std::vector<size_t>>{{9}, {1, 7, 31}, {4095, 4096, 4097}, {4608, 76032, 16896}}) {
        for (size_t gap : {size_t(0), size_t(3)}) {
            Fixture f(sizes, gap);
            if (sizes == std::vector<size_t>{9} && gap == 0) {
                ASSERT_EQ(cudaMemcpy(f.device, "123456789", 9, cudaMemcpyHostToDevice), cudaSuccess);
            }
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            const auto original = f.deviceBytes();
            ASSERT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, false, f.request_id).success);
            EXPECT_EQ(f.footer().crc32c, crc32c(f.host, f.bytes));
            if (sizes == std::vector<size_t>{9} && gap == 0) {
                EXPECT_EQ(f.footer().crc32c, 0xe3069283u);
            }
            ASSERT_EQ(cudaMemset(f.device, 0xa5, f.bytes), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            ASSERT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, true, f.request_id).success);
            const auto restored = f.deviceBytes();
            for (const auto& tile : f.tiles)
                EXPECT_EQ(std::memcmp(original.data() + tile.offset, restored.data() + tile.offset, tile.bytes), 0);
        }
    }
}

TEST(CrcBlockCopyTest, PayloadAndStoredCrcTamperingRejectBeforeScatter) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    for (bool corrupt_footer : {false, true}) {
        SCOPED_TRACE(corrupt_footer);
        Fixture f({19, 4096, 31});
        f.copy->gather(f.bytes, f.tiles);
        ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id).success);
        const auto stored_crc = f.footer().crc32c;
        ASSERT_EQ(cudaMemset(f.device, 0xa5, f.bytes), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        const auto untouched = f.deviceBytes();
        if (corrupt_footer) {
            f.host[CrcBlockCopy::footerOffset(f.bytes)] ^= 1;
        } else {
            f.host[f.bytes - 1] ^= 1;
        }
        const auto rejected = f.copy->loadAndValidate(f.host, f.bytes, f.pinned, [] { return true; });
        EXPECT_FALSE(rejected.success);
        EXPECT_EQ(rejected.failure_stage, CrcBlockCopyResult::FailureStage::SOURCE_CRC);
        EXPECT_TRUE(rejected.gpu_crc_observed);
        EXPECT_EQ(rejected.gpu_crc_status, 0u);
        EXPECT_NE(rejected.expected_crc, rejected.actual_crc);
        EXPECT_EQ(rejected.actual_crc, crc32c(f.host, f.bytes));
        EXPECT_EQ(rejected.expected_crc, f.footer().crc32c);
        if (corrupt_footer) {
            EXPECT_EQ(rejected.actual_crc, stored_crc);
        }
        if (rejected.success)
            f.copy->scatter(f.bytes, f.tiles);
        EXPECT_EQ(untouched, f.deviceBytes());
        ASSERT_EQ(rejected.staging_snapshot.size(), f.bytes);
        EXPECT_EQ(std::memcmp(rejected.staging_snapshot.data(), f.host, f.bytes), 0);
    }
}

TEST(CrcBlockCopyTest, WriterProvenanceIsOutsideTransferAndCrc) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    ASSERT_EQ(sizeof(CrcBlockFooter), 4u);
    ASSERT_EQ(sizeof(CrcBlockHostMetadata), 8u);
    for (bool pinned : {false, true}) {
        SCOPED_TRACE(pinned);
        Fixture    f({9}, 0, true, pinned);
        const auto original = f.deviceBytes();
        ASSERT_EQ(CrcBlockCopy::footerOffset(f.bytes), 16u);
        ASSERT_EQ(CrcBlockCopy::transferBytes(f.bytes), 20u);
        ASSERT_EQ(CrcBlockCopy::storageBytes(f.bytes), 32u);
        f.copy->gather(f.bytes, f.tiles);
        ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id, pinned).success);
        EXPECT_EQ(f.writerRequestId(), f.request_id);
        const auto    stored_crc   = f.footer().crc32c;
        const int64_t other_writer = -123456789012345LL;
        std::memcpy(f.host + CrcBlockCopy::transferBytes(f.bytes), &other_writer, sizeof(other_writer));
        std::fill(f.host + CrcBlockCopy::transferBytes(f.bytes) + sizeof(CrcBlockHostMetadata),
                  f.host + CrcBlockCopy::storageBytes(f.bytes),
                  0x6c);
        ASSERT_EQ(cudaMemset(f.device, 0xa5, f.bytes), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        const auto untouched = f.deviceBytes();
        const auto loaded    = f.copy->loadAndValidate(f.host, f.bytes, pinned);
        ASSERT_TRUE(loaded.success);
        EXPECT_EQ(loaded.actual_crc, stored_crc);
        EXPECT_EQ(untouched, f.deviceBytes());  // load returns the CPU decision without scattering.
        f.copy->scatter(f.bytes, f.tiles);
        EXPECT_EQ(original, f.deviceBytes());
        EXPECT_EQ(f.writerRequestId(), other_writer);
        f.copy->gather(f.bytes, f.tiles);
        ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id, pinned).success);
        EXPECT_EQ(f.writerRequestId(), f.request_id);
        EXPECT_EQ(f.footer().crc32c, stored_crc);
        for (size_t i = CrcBlockCopy::transferBytes(f.bytes) + sizeof(CrcBlockHostMetadata);
             i < CrcBlockCopy::storageBytes(f.bytes);
             ++i)
            EXPECT_EQ(f.host[i], 0);
    }

    // Make provenance inaccessible: load needs only payload padding plus the four-byte CRC.
    Fixture f({9});
    f.copy->gather(f.bytes, f.tiles);
    ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id).success);
    const auto original  = f.deviceBytes();
    const long page_size = ::sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    ASSERT_GT(static_cast<size_t>(page_size), CrcBlockCopy::transferBytes(f.bytes));
    void* mapping = ::mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto                                   unmap = [page_size](void* ptr) { ::munmap(ptr, 2 * page_size); };
    std::unique_ptr<void, decltype(unmap)> pages(mapping, unmap);
    auto*                                  boundary    = static_cast<uint8_t*>(mapping) + page_size;
    auto*                                  transferred = boundary - CrcBlockCopy::transferBytes(f.bytes);
    std::memcpy(transferred, f.host, CrcBlockCopy::transferBytes(f.bytes));
    ASSERT_EQ(::mprotect(boundary, page_size, PROT_NONE), 0);
    ASSERT_EQ(cudaMemset(f.device, 0xa5, f.bytes), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    ASSERT_TRUE(f.copy->loadAndValidate(transferred, f.bytes, false).success);
    f.copy->scatter(f.bytes, f.tiles);
    EXPECT_EQ(original, f.deviceBytes());
}

TEST(CrcBlockCopyTest, InvalidPayloadAndTileBoundsThrowBeforeCopy) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    EXPECT_THROW(CrcBlockCopy(std::vector<size_t>{}, 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopy(std::vector<size_t>{12}, 0), std::invalid_argument);
    Fixture f({4, 4, 4});
    f.copy->gather(f.bytes, f.tiles);
    ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id).success);
    ASSERT_TRUE(f.copy->loadAndValidate(f.host, f.bytes).success);
    const auto                                       original = f.deviceBytes();
    const std::vector<uint8_t>                       host_before(f.host, f.host + CrcBlockCopy::storageBytes(f.bytes));
    const std::vector<std::vector<CrcBlockCopyTile>> invalid_tiles{
        {},
        {{nullptr, 0, 1}},
        {{f.device, 0, 0}},
        {{f.device, f.bytes, 1}},
        {{f.device, f.bytes - 1, 2}},
        {{f.device, std::numeric_limits<size_t>::max(), 1}},
        {{f.device, 1, std::numeric_limits<size_t>::max()}},
        {{f.device, 0, 2}, {f.device + 1, 1, 2}},
        {{f.device + 2, 2, 1}, {f.device, 0, 1}},
        {{f.device, 0, 1}, {f.device + 1, 1, 1}, {f.device + 2, 2, 1}, {f.device + 3, 3, 1}},
    };
    for (size_t i = 0; i < invalid_tiles.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_THROW(f.copy->gather(f.bytes, invalid_tiles[i]), std::invalid_argument);
        EXPECT_THROW(f.copy->scatter(f.bytes, invalid_tiles[i]), std::invalid_argument);
    }
    for (size_t bytes : {size_t(0), f.bytes + 1}) {
        SCOPED_TRACE(bytes);
        EXPECT_THROW(f.copy->gather(bytes, f.tiles), std::invalid_argument);
        EXPECT_THROW(f.copy->store(f.host, bytes), std::invalid_argument);
        EXPECT_THROW(f.copy->loadAndValidate(f.host, bytes), std::invalid_argument);
        EXPECT_THROW(f.copy->scatter(bytes, f.tiles), std::invalid_argument);
    }
    EXPECT_THROW(f.copy->store(nullptr, f.bytes), std::invalid_argument);
    EXPECT_THROW(f.copy->loadAndValidate(nullptr, f.bytes), std::invalid_argument);
    EXPECT_EQ(original, f.deviceBytes());
    EXPECT_EQ(std::memcmp(f.host, host_before.data(), host_before.size()), 0);
    ASSERT_TRUE(f.copy->loadAndValidate(f.host, f.bytes).success);
    f.copy->scatter(f.bytes, f.tiles);
    EXPECT_EQ(original, f.deviceBytes());
}

TEST(CrcBlockCopyTest, DumpAllocationFailurePreservesRejection) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    Fixture f({4096});
    f.copy->gather(f.bytes, f.tiles);
    ASSERT_TRUE(f.copy->store(f.host, f.bytes, f.request_id).success);
    f.host[0] ^= 1;
    CrcBlockCopyResult result;
    EXPECT_NO_THROW(result =
                        f.copy->loadAndValidate(f.host, f.bytes, f.pinned, []() -> bool { throw std::bad_alloc(); }));
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.gpu_crc_observed);
    EXPECT_TRUE(result.staging_snapshot.empty());
}

TEST(CrcBlockCopyTest, InheritVerifiedBlockAndRejectDirtySourceWithoutChangingCandidate) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    Fixture source({4096, 4096});
    Fixture candidate({4096, 4096});
    ASSERT_TRUE(copyBlock(*source.copy, source.host, source.bytes, source.tiles, false, source.request_id).success);
    ++candidate.request_id;
    ASSERT_EQ(cudaMemset(candidate.device, 0x7d, candidate.bytes), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const std::vector<CrcBlockCopyTile> tail{candidate.tiles[1]};
    ASSERT_TRUE(
        copyBlock(*candidate.copy, candidate.host, candidate.bytes, tail, false, candidate.request_id, source.host)
            .success);
    EXPECT_EQ(std::memcmp(candidate.host, source.host, 4096), 0);
    for (size_t i = 4096; i < candidate.bytes; ++i)
        EXPECT_EQ(candidate.host[i], 0x7d);
    EXPECT_EQ(candidate.footer().crc32c, crc32c(candidate.host, candidate.bytes));
    EXPECT_EQ(candidate.writerRequestId(), candidate.request_id);
    const std::vector<uint8_t> before(candidate.host, candidate.host + CrcBlockCopy::storageBytes(candidate.bytes));
    source.host[17] ^= 1;
    const auto rejected = copyBlock(
        *candidate.copy, candidate.host, candidate.bytes, tail, false, candidate.request_id + 1, source.host, [] {
            return true;
        });
    EXPECT_FALSE(rejected.success);
    EXPECT_FALSE(rejected.output_written);
    EXPECT_EQ(std::memcmp(candidate.host, before.data(), before.size()), 0);
    ASSERT_EQ(rejected.staging_snapshot.size(), source.bytes);
    EXPECT_EQ(std::memcmp(rejected.staging_snapshot.data(), source.host, source.bytes), 0);
    ASSERT_TRUE(copyBlock(*candidate.copy, candidate.host, candidate.bytes, candidate.tiles, true).success);
}

TEST(CrcBlockCopyTest, PageableBounceRoundtripAndMixedSource) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    for (bool source_pinned : {false, true}) {
        Fixture source({31, 4096}, 0, true, source_pinned);
        Fixture target({31, 4096}, 0, true, false);
        ASSERT_TRUE(copyBlock(*source.copy,
                              source.host,
                              source.bytes,
                              source.tiles,
                              false,
                              source.request_id,
                              nullptr,
                              {},
                              source.pinned)
                        .success);
        ++target.request_id;
        ASSERT_EQ(cudaMemset(target.device, 0x7d, target.bytes), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        ASSERT_TRUE(copyBlock(*target.copy,
                              target.host,
                              target.bytes,
                              {target.tiles.back()},
                              false,
                              target.request_id,
                              source.host,
                              {},
                              false,
                              source.pinned)
                        .success);
        EXPECT_EQ(std::memcmp(target.host, source.host, 31), 0);
        for (size_t i = 31; i < target.bytes; ++i) {
            EXPECT_EQ(target.host[i], 0x7d);
        }
        EXPECT_EQ(target.footer().crc32c, crc32c(target.host, target.bytes));
        ASSERT_EQ(cudaMemset(target.device, 0xa5, target.bytes), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        ASSERT_TRUE(
            copyBlock(
                *target.copy, target.host, target.bytes, target.tiles, true, target.request_id, nullptr, {}, false)
                .success);
        EXPECT_EQ(std::memcmp(target.deviceBytes().data(), target.host, target.bytes), 0);
        target.host[0] ^= 1;
        EXPECT_FALSE(
            copyBlock(
                *target.copy, target.host, target.bytes, target.tiles, true, target.request_id, nullptr, {}, false)
                .success);
    }
}

TEST(CrcBlockCopyTest, ReusedWorkspaceClearsUnwrittenSlots) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    Fixture f({4096, 4096, 4096});
    f.copy              = std::make_unique<CrcBlockCopy>(std::vector<size_t>{f.bytes, 9}, f.tiles.size());
    const auto original = f.deviceBytes();
    for (const auto& indices : std::vector<std::vector<size_t>>{{2}, {0, 2}, {0}}) {
        ++f.request_id;
        ASSERT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, false, f.request_id).success);
        std::vector<CrcBlockCopyTile> partial;
        std::vector<uint8_t>          expected(f.bytes, 0);
        for (const auto index : indices) {
            const auto& tile = f.tiles[index];
            partial.push_back(tile);
            std::memcpy(expected.data() + tile.offset, original.data() + tile.offset, tile.bytes);
        }
        ++f.request_id;
        ASSERT_TRUE(copyBlock(*f.copy, f.host, f.bytes, partial, false, f.request_id).success);
        EXPECT_EQ(std::memcmp(f.host, expected.data(), f.bytes), 0);
        EXPECT_EQ(f.footer().crc32c, crc32c(f.host, f.bytes));
    }

    // A smaller, fully covered payload must not leak the previous payload into its alignment padding.
    ++f.request_id;
    ASSERT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, false, f.request_id).success);
    ++f.request_id;
    ASSERT_TRUE(copyBlock(*f.copy, f.host, 9, {{f.device, 0, 9}}, false, f.request_id).success);
    EXPECT_EQ(std::memcmp(f.host, original.data(), 9), 0);
    for (size_t i = 9; i < CrcBlockCopy::footerOffset(9); ++i)
        EXPECT_EQ(f.host[i], 0);
    CrcBlockFooter footer;
    std::memcpy(&footer, f.host + CrcBlockCopy::footerOffset(9), sizeof(footer));
    EXPECT_EQ(footer.crc32c, crc32c(f.host, 9));
}

TEST(CrcBlockCopyTest, CpuFixedPoolDmaRoundtripAndRejectedInheritancePreserveEvidence) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    for (size_t host_count : {size_t(1), size_t(8)}) {
        SCOPED_TRACE(host_count);
        for (bool pin_backing : {false, true}) {
            SCOPED_TRACE(pin_backing);
            std::vector<size_t> sizes{31}, host_indices;
            for (size_t i = 0; i < host_count; ++i) {
                host_indices.push_back(sizes.size());
                sizes.push_back(4096 / host_count);
                if (host_count > 1 && i % 2 == 0)
                    sizes.push_back(17);  // GPU regions make consecutive CPU pitches alternate.
            }
            sizes.push_back(17);
            Fixture  source(sizes, 3, true, pin_backing);
            uint8_t* cpu_address = nullptr;
            ASSERT_EQ(cudaMallocHost(&cpu_address, 4096), cudaSuccess);
            auto release_cpu = [](uint8_t* pointer) { cudaFreeHost(pointer); };
            std::unique_ptr<uint8_t, decltype(release_cpu)> cpu_pool(cpu_address, release_cpu);
            for (size_t i = 0; i < 4096; ++i)
                cpu_address[i] = (i * 131 + 17 * (i >> 8) + 0x2e) & 255;
            for (size_t i = 0; i < host_count; ++i) {
                source.tiles[host_indices[i]].address = cpu_address + i * (4096 / host_count);
                source.tiles[host_indices[i]].is_cuda = false;
            }
            const auto           gpu_source = source.deviceBytes();
            std::vector<uint8_t> original(source.bytes, 0);
            for (const auto& tile : source.tiles) {
                const auto* data =
                    tile.is_cuda ? gpu_source.data() + tile.offset : static_cast<const uint8_t*>(tile.address);
                std::memcpy(original.data() + tile.offset, data, tile.bytes);
            }
            ASSERT_TRUE(copyBlock(*source.copy,
                                  source.host,
                                  source.bytes,
                                  source.tiles,
                                  false,
                                  source.request_id,
                                  nullptr,
                                  {},
                                  pin_backing)
                            .success);
            EXPECT_EQ(std::memcmp(source.host, original.data(), original.size()), 0);
            EXPECT_EQ(source.footer().crc32c, crc32c(source.host, source.bytes));
            ASSERT_EQ(cudaMemset(source.device, 0xa5, source.bytes), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            std::memset(cpu_address, 0xa5, 4096);
            ASSERT_TRUE(copyBlock(*source.copy,
                                  source.host,
                                  source.bytes,
                                  source.tiles,
                                  true,
                                  source.request_id,
                                  nullptr,
                                  {},
                                  pin_backing)
                            .success);
            auto restored = source.deviceBytes();
            for (const auto& tile : source.tiles) {
                const auto* target =
                    tile.is_cuda ? restored.data() + tile.offset : static_cast<const uint8_t*>(tile.address);
                EXPECT_EQ(std::memcmp(target, original.data() + tile.offset, tile.bytes), 0);
            }

            source.host[source.bytes - 1] ^= 1;
            std::memset(cpu_address, 0x6c, 4096);
            const std::vector<uint8_t> cpu_before(cpu_address, cpu_address + 4096);
            const auto                 bad_read = copyBlock(*source.copy,
                                            source.host,
                                            source.bytes,
                                            source.tiles,
                                            true,
                                            source.request_id,
                                            nullptr,
                                                            {},
                                            pin_backing);
            EXPECT_FALSE(bad_read.success);
            EXPECT_EQ(source.deviceBytes(), restored);
            EXPECT_EQ(std::memcmp(cpu_address, cpu_before.data(), cpu_before.size()), 0);

            Fixture candidate(sizes, 3, true, pin_backing);
            ++candidate.request_id;
            std::memset(candidate.host, 0x3b, CrcBlockCopy::storageBytes(candidate.bytes));
            const std::vector<uint8_t> candidate_before(candidate.host,
                                                        candidate.host + CrcBlockCopy::storageBytes(candidate.bytes));
            std::memset(cpu_address, 0x77, 4096);
            std::vector<CrcBlockCopyTile> host_overlay;
            for (const auto index : host_indices)
                host_overlay.push_back(source.tiles[index]);
            auto rejected = copyBlock(
                *candidate.copy,
                candidate.host,
                candidate.bytes,
                host_overlay,
                false,
                candidate.request_id,
                source.host,
                [] { return true; },
                pin_backing,
                pin_backing);
            EXPECT_FALSE(rejected.success);
            EXPECT_FALSE(rejected.output_written);
            EXPECT_EQ(std::memcmp(candidate.host, candidate_before.data(), candidate_before.size()), 0);
            ASSERT_EQ(rejected.staging_snapshot.size(), source.bytes);
            EXPECT_EQ(std::memcmp(rejected.staging_snapshot.data(), source.host, source.bytes), 0);

            source.host[source.bytes - 1] ^= 1;
            ASSERT_TRUE(copyBlock(*candidate.copy,
                                  candidate.host,
                                  candidate.bytes,
                                  host_overlay,
                                  false,
                                  candidate.request_id,
                                  source.host,
                                  {},
                                  pin_backing,
                                  pin_backing)
                            .success);
            auto expected = original;
            for (const auto& tile : host_overlay)
                std::fill(expected.begin() + tile.offset, expected.begin() + tile.offset + tile.bytes, 0x77);
            EXPECT_EQ(std::memcmp(candidate.host, expected.data(), expected.size()), 0);
            EXPECT_EQ(candidate.footer().crc32c, crc32c(candidate.host, candidate.bytes));
        }
    }
}

class Barrier {
    std::mutex              mutex_;
    std::condition_variable cv_;
    int                     total_, arrived_{0}, generation_{0};

public:
    explicit Barrier(int n): total_(n) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const int                    gen = generation_;
        if (++arrived_ == total_) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
        } else
            cv_.wait(lock, [&] { return generation_ != gen; });
    }
};

std::vector<size_t> proLayout(size_t scale) {
    std::vector<size_t> result;
    for (int layer = 0; layer < 61; ++layer) {
        if (layer < 2 || layer % 2)
            result.push_back(1152 * scale);
        else {
            result.push_back(19008 * scale);
            result.push_back(4224 * scale);
        }
    }
    return result;
}

std::vector<size_t> stateLayout() {
    // DSV4 Pro FP8, CP=1, no MTP slack: 8 FP32 state entries; 128 SWA entries.
    std::vector<size_t> result;
    for (int layer = 0; layer < 61; ++layer) {
        if (layer >= 2 && layer % 2 == 0) {
            result.push_back(16384);
            result.push_back(65536);
        }
        result.push_back(74880);  // align_up(128 * 584, 576)
    }
    return result;
}

TEST(CrcBlockCopyTest, EightWorkspacesKeepDifferentBlocksIsolated) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    std::vector<std::unique_ptr<Fixture>> fixtures;
    for (int w = 0; w < 8; ++w) {
        auto f = std::make_unique<Fixture>(std::vector<size_t>{33, 4096, 17});
        f->request_id += w;
        for (size_t i = 0; i < f->bytes; ++i)
            f->host[i] = (i * 17 + w * 29) & 255;
        ASSERT_EQ(cudaMemcpy(f->device, f->host, f->bytes, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        fixtures.push_back(std::move(f));
    }
    Barrier                  barrier(8);
    std::vector<std::thread> threads;
    for (int w = 0; w < 8; ++w)
        threads.emplace_back([&, w] {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            auto& f = *fixtures[w];
            for (int run = 0; run < 20; ++run) {
                barrier.wait();
                EXPECT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, false, f.request_id).success);
                EXPECT_EQ(f.footer().crc32c, crc32c(f.host, f.bytes));
                EXPECT_EQ(f.writerRequestId(), f.request_id);
                EXPECT_TRUE(copyBlock(*f.copy, f.host, f.bytes, f.tiles, true, f.request_id).success);
                const auto data = f.deviceBytes();
                for (size_t i = 0; i < f.bytes; ++i)
                    EXPECT_EQ(data[i], (i * 17 + w * 29) & 255);
            }
        });
    for (auto& thread : threads)
        thread.join();
}

TEST(CrcBlockCopyTest, BenchmarkOriginalExecutors) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    if (!std::getenv("RTP_LLM_CRC_BENCHMARK")) {
        GTEST_SKIP() << "opt-in performance benchmark";
    }
    struct Case {
        const char*         name;
        std::vector<size_t> sizes;
        int                 blocks;
        size_t              active_tiles{0};
        bool                inherit{false};
        size_t              only_tile_bytes{0};
    };
    const std::vector<Case> cases{
        {"compressed128", proLayout(1), 1},
        {"compressed512", proLayout(4), 1},
        {"compressed512_batch8", proLayout(4), 8},
        {"state_swa", stateLayout(), 1},
        {"compressed128_sparse1", proLayout(1), 1, 1},
        {"compressed512_sparse1", proLayout(4), 1, 1},
        {"state_swa_sparse1", stateLayout(), 1, 1},
        {"compressed128_merge1", proLayout(1), 1, 1, true},
        {"compressed512_merge1", proLayout(4), 1, 1, true},
        {"state_swa_merge1", stateLayout(), 1, 1, true},
        {"compressed128_hca_only", proLayout(1), 1, 31, false, 1152},
        {"compressed512_hca_only", proLayout(4), 1, 31, false, 4608},
        {"compressed512_hca_merge", proLayout(4), 1, 31, true, 4608},
        {"state_swa_swa_only", stateLayout(), 1, 61, false, 74880},
        {"state_swa_indexer_state_only", stateLayout(), 1, 30, false, 16384},
        {"state_swa_indexer_state_merge", stateLayout(), 1, 30, true, 16384},
    };
    for (const auto& test : cases) {
        const auto* selected_case = std::getenv("RTP_LLM_CRC_BENCH_CASE");
        if (selected_case && std::strcmp(test.name, selected_case) != 0)
            continue;
        if (bool(test.active_tiles) != bool(std::getenv("RTP_LLM_CRC_BENCH_SPARSE")))
            continue;
        for (int concurrency : {1, 8}) {
            const int                                          rounds = test.blocks > 1 ? 100 : 300;
            std::vector<std::vector<std::unique_ptr<Fixture>>> fixtures(concurrency), sources(concurrency);
            for (int w = 0; w < concurrency; ++w)
                for (int block = 0; block < test.blocks; ++block) {
                    auto f = std::make_unique<Fixture>(
                        test.sizes, 0, block == 0, !std::getenv("RTP_LLM_CRC_BENCH_PAGEABLE"));
                    f->request_id += w * test.blocks + block;
                    for (size_t i = 0; i < f->bytes; ++i)
                        f->host[i] = (i * 131 + w * 17 + block * 31) & 255;
                    ASSERT_EQ(cudaMemcpy(f->device, f->host, f->bytes, cudaMemcpyHostToDevice), cudaSuccess);
                    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
                    fixtures[w].push_back(std::move(f));
                    auto& item = *fixtures[w].back();
                    ASSERT_TRUE(copyBlock(*fixtures[w][0]->copy,
                                          item.host,
                                          item.bytes,
                                          item.tiles,
                                          false,
                                          item.request_id,
                                          nullptr,
                                          {},
                                          item.pinned)
                                    .success);
                    if (test.inherit) {
                        auto source        = std::make_unique<Fixture>(test.sizes, 0, false, item.pinned);
                        source->request_id = item.request_id;
                        ASSERT_TRUE(copyBlock(*fixtures[w][0]->copy,
                                              source->host,
                                              source->bytes,
                                              source->tiles,
                                              false,
                                              source->request_id,
                                              nullptr,
                                              {},
                                              source->pinned)
                                        .success);
                        sources[w].push_back(std::move(source));
                        ++item.request_id;
                    }
                    if (test.only_tile_bytes) {
                        item.tiles.erase(
                            std::remove_if(item.tiles.begin(),
                                           item.tiles.end(),
                                           [&](const auto& tile) { return tile.bytes != test.only_tile_bytes; }),
                            item.tiles.end());
                        ASSERT_EQ(item.tiles.size(), test.active_tiles);
                    } else if (test.active_tiles) {
                        item.tiles.resize(test.active_tiles);
                    }
                }
            const size_t            block_bytes = fixtures[0][0]->bytes;
            StagedMemoryCopyScratch original_scratch;
            std::mutex              original_mutex;
            // Sparse prefix items use the original generic executor, not the legacy staged path.
            std::vector<int> modes = test.active_tiles ? std::vector<int>{0, 2} : std::vector<int>{0, 1, 2};
            if (std::getenv("RTP_LLM_CRC_BENCH_REVERSE"))
                std::reverse(modes.begin(), modes.end());
            for (bool h2d : {false, true}) {
                if (h2d && test.inherit)
                    continue;
                for (int mode : modes) {
                    Barrier                          barrier(concurrency);
                    std::vector<std::vector<double>> samples(concurrency);
                    std::vector<std::thread>         workers;
                    for (int w = 0; w < concurrency; ++w)
                        workers.emplace_back([&, w] {
                            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
                            // All paths receive prebuilt descriptors: these are executor timings.
                            std::vector<MultiCopyParams> generic(test.blocks);
                            StagedMemoryCopyParams       staged;
                            staged.host_bytes   = block_bytes * test.blocks;
                            staged.device_index = 0;
                            staged.direction    = h2d ? StagedMemoryCopyDirection::H2D : StagedMemoryCopyDirection::D2H;
                            for (int block = 0; block < test.blocks; ++block) {
                                const auto& f = *fixtures[w][block];
                                for (const auto& tile : f.tiles) {
                                    auto cpu = torch::from_blob(f.host + tile.offset,
                                                                {static_cast<int64_t>(tile.bytes)},
                                                                torch::TensorOptions().dtype(torch::kUInt8));
                                    auto gpu = torch::from_blob(
                                        tile.address,
                                        {static_cast<int64_t>(tile.bytes)},
                                        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
                                    generic[block].multi_src.push_back(h2d ? cpu : gpu);
                                    generic[block].multi_dst.push_back(h2d ? gpu : cpu);
                                    const size_t offset = block * block_bytes + tile.offset;
                                    staged.tiles.push_back({tile.address, offset, tile.bytes});
                                    staged.host_segments.push_back({f.host + tile.offset, offset, tile.bytes});
                                }
                            }
                            auto run = [&] {
                                if (mode == 2) {
                                    for (int block = 0; block < test.blocks; ++block) {
                                        auto& item = fixtures[w][block];
                                        EXPECT_TRUE(copyBlock(*fixtures[w][0]->copy,
                                                              item->host,
                                                              item->bytes,
                                                              item->tiles,
                                                              h2d,
                                                              item->request_id,
                                                              test.inherit ? sources[w][block]->host : nullptr,
                                                              {},
                                                              item->pinned,
                                                              test.inherit ? sources[w][block]->pinned : true)
                                                        .success);
                                    }
                                } else if (mode == 1) {
                                    std::lock_guard<std::mutex> lock(original_mutex);
                                    EXPECT_TRUE(execStagedMemoryCopy(staged, &original_scratch));
                                } else {
                                    for (int block = 0; block < test.blocks; ++block) {
                                        if (test.inherit) {
                                            std::memcpy(fixtures[w][block]->host, sources[w][block]->host, block_bytes);
                                        }
                                        execNoBlockCopy(generic[block]);
                                    }
                                }
                            };
                            for (int i = 0; i < 20; ++i)
                                run();
                            for (int i = 0; i < rounds; ++i) {
                                barrier.wait();
                                const auto before = std::chrono::steady_clock::now();
                                run();
                                samples[w].push_back(
                                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - before)
                                        .count());
                            }
                        });
                    for (auto& worker : workers)
                        worker.join();
                    std::vector<double> all;
                    for (auto& part : samples)
                        all.insert(all.end(), part.begin(), part.end());
                    std::sort(all.begin(), all.end());
                    std::printf(
                        "CRC_COPY_BENCH case=%s direction=%s mode=%d concurrency=%d blocks=%d block_bytes=%zu n=%zu pinned=%d p50_us=%.3f p99_us=%.3f\n",
                        test.name,
                        h2d ? "H2D" : "D2H",
                        mode,
                        concurrency,
                        test.blocks,
                        block_bytes,
                        all.size(),
                        fixtures[0][0]->pinned ? 1 : 0,
                        all[all.size() / 2],
                        all[(all.size() - 1) * 99 / 100]);
                }
            }
            releaseStagedMemoryCopyScratch(original_scratch);
        }
    }
}

}  // namespace
}  // namespace rtp_llm::test
