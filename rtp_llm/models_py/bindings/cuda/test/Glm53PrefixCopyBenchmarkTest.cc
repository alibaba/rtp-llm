#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {
namespace {

double threadMicros() {
    timespec value{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
    return value.tv_sec * 1e6 + value.tv_nsec * 1e-3;
}

struct Timing {
    double wall_us;
    double cpu_us;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

class Glm53PrefixCopyBenchmarkTest: public ::testing::Test {
protected:
    // Model trace: each of 352 compressed blocks has eleven MLA layers,
    // each restoring these four regions. Linear state is one separate 36 MiB copy.
    static constexpr size_t kLayers       = 11;
    static constexpr size_t kGap          = 256;
    static constexpr size_t kBlockBytes   = kLayers * (131072 + 131072 + 4224 + 8320);
    uint8_t*                host_         = nullptr;
    uint8_t*                device_       = nullptr;
    size_t                  device_bytes_ = 0;
    std::vector<size_t>     device_offsets_;
    std::vector<size_t>     host_offsets_;
    BatchedMemoryCopyParams batch_;
    StagedMemoryCopyParams  staged_;
    StagedMemoryCopyScratch scratch_;

    void allocate(size_t blocks) {
        ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&host_), blocks * kBlockBytes, cudaHostAllocPortable),
                  cudaSuccess);
        device_bytes_ = blocks * (kBlockBytes + kLayers * 4 * kGap);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_), device_bytes_), cudaSuccess);
        batch_.device_index                 = 0;
        staged_.device_index                = 0;
        staged_.direction                   = StagedMemoryCopyDirection::H2D;
        staged_.direct_pinned_host_segments = true;
        staged_.host_bytes                  = blocks * kBlockBytes;
        size_t host_offset = 0, device_offset = 0;
        for (size_t block = 0; block < blocks; ++block) {
            staged_.host_segments.push_back({host_ + host_offset, host_offset, kBlockBytes});
            for (size_t layer = 0; layer < kLayers; ++layer) {
                for (size_t bytes : {131072u, 131072u, 4224u, 8320u}) {
                    batch_.tiles.push_back({device_ + device_offset, host_ + host_offset, bytes});
                    staged_.tiles.push_back({device_ + device_offset, host_offset, bytes});
                    host_offsets_.push_back(host_offset);
                    device_offsets_.push_back(device_offset);
                    host_offset += bytes;
                    device_offset += bytes + kGap;
                }
            }
        }
    }

    bool copyBounded(size_t capacity, int sm_blocks, StagedMemoryCopyScratch& scratch) {
        // Test the complete planning/submission cost with a bounded workspace.
        // Keep one host segment per compressed block, even if this fixture's
        // allocations happen to be adjacent, as in the unbounded baseline.
        StagedMemoryCopyParams part;
        part.device_index                = 0;
        part.direction                   = StagedMemoryCopyDirection::H2D;
        part.direct_pinned_host_segments = true;
        part.sm_copy_block_num           = sm_blocks;
        part.tiles.reserve(1024);
        part.host_segments.reserve(32);
        auto flush = [&]() {
            if (!execStagedMemoryCopy(part, &scratch))
                return false;
            part.tiles.clear();
            part.host_segments.clear();
            part.host_bytes = 0;
            return true;
        };
        for (size_t i = 0; i < batch_.tiles.size(); ++i) {
            const auto& tile = batch_.tiles[i];
            if (tile.bytes > capacity)
                return false;
            if (part.host_bytes + tile.bytes > capacity && !flush())
                return false;
            auto* host = const_cast<void*>(tile.src);
            if (!part.host_segments.empty() && i % (kLayers * 4) != 0) {
                auto& last = part.host_segments.back();
                if (static_cast<char*>(last.host) + last.bytes == host
                    && last.host_offset + last.bytes == part.host_bytes) {
                    last.bytes += tile.bytes;
                } else {
                    part.host_segments.push_back({host, part.host_bytes, tile.bytes});
                }
            } else {
                part.host_segments.push_back({host, part.host_bytes, tile.bytes});
            }
            part.tiles.push_back({tile.dst, part.host_bytes, tile.bytes});
            part.host_bytes += tile.bytes;
        }
        return flush();
    }

    void fill(uint8_t seed) {
        for (size_t i = 0; i < batch_.tiles.size(); ++i) {
            const auto& tile = batch_.tiles[i];
            auto*       ptr  = host_ + host_offsets_[i];
            std::memset(ptr, seed + i % 251, tile.bytes);
            // Unique markers catch mistaken region order despite byte-pattern wrap.
            const uint64_t marker = i * 1009 + seed;
            std::memcpy(ptr, &marker, sizeof(marker));
            std::memcpy(ptr + tile.bytes - sizeof(marker), &marker, sizeof(marker));
        }
    }

    void clear() {
        ASSERT_EQ(cudaMemset(device_, 0xA5, device_bytes_), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    void verify() {
        std::vector<uint8_t> actual(device_bytes_);
        ASSERT_EQ(cudaMemcpy(actual.data(), device_, device_bytes_, cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<uint8_t> gap(kGap, 0xA5);
        for (size_t i = 0; i < batch_.tiles.size(); ++i) {
            const auto bytes = batch_.tiles[i].bytes;
            ASSERT_EQ(std::memcmp(actual.data() + device_offsets_[i], host_ + host_offsets_[i], bytes), 0) << i;
            ASSERT_EQ(std::memcmp(actual.data() + device_offsets_[i] + bytes, gap.data(), kGap), 0) << i;
        }
    }

    void release() {
        releaseStagedMemoryCopyScratch(scratch_);
        if (device_) {
            EXPECT_EQ(cudaFree(device_), cudaSuccess);
            device_ = nullptr;
        }
        if (host_) {
            EXPECT_EQ(cudaFreeHost(host_), cudaSuccess);
            host_ = nullptr;
        }
        batch_  = {};
        staged_ = {};
        device_offsets_.clear();
        host_offsets_.clear();
    }
    void TearDown() override {
        release();
    }
};

TEST_F(Glm53PrefixCopyBenchmarkTest, ExactRoundTripAndCpuCost) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    unsigned original_flags = 0;
    ASSERT_EQ(cudaGetDeviceFlags(&original_flags), cudaSuccess);
    struct RestoreFlags {
        unsigned flags;
        ~RestoreFlags() {
            (void)cudaSetDeviceFlags(flags);
        }
    } restore{original_flags};
    ASSERT_EQ(cudaSetDeviceFlags((original_flags & ~cudaDeviceScheduleMask) | cudaDeviceScheduleBlockingSync),
              cudaSuccess);
    for (size_t blocks : {1u, 32u, 352u}) {
        SCOPED_TRACE(blocks);
        allocate(blocks);
        fill(19);
        clear();
        ASSERT_TRUE(execBatchedMemoryCopy(batch_));
        verify();
        // Check each grid variant, reused scratch, changed contents and gaps.
        for (int sm_blocks : {0, 16, 64, 256}) {
            staged_.sm_copy_block_num = sm_blocks;
            fill(static_cast<uint8_t>(31 + sm_blocks));
            clear();
            ASSERT_TRUE(execStagedMemoryCopy(staged_, &scratch_));
            verify();
            ASSERT_EQ(scratch_.host_allocation_count, 0u);
        }
        std::vector<std::vector<Timing>> timings(5);
        const int                        grid[] = {-1, 0, 16, 64, 256};
        for (int variant = 0; variant < 5; ++variant) {
            staged_.sm_copy_block_num = std::max(0, grid[variant]);
            for (int warm = 0; warm < 30; ++warm) {
                ASSERT_TRUE(variant == 0 ? execBatchedMemoryCopy(batch_) : execStagedMemoryCopy(staged_, &scratch_));
            }
        }
        for (int repeat = 0; repeat < 60; ++repeat) {
            for (int n = 0; n < 5; ++n) {
                const int variant         = repeat % 2 ? n : 4 - n;
                staged_.sm_copy_block_num = std::max(0, grid[variant]);
                const auto   start        = std::chrono::steady_clock::now();
                const double cpu_start    = threadMicros();
                const bool ok = variant == 0 ? execBatchedMemoryCopy(batch_) : execStagedMemoryCopy(staged_, &scratch_);
                const double cpu_us = threadMicros() - cpu_start;
                const auto   end    = std::chrono::steady_clock::now();
                ASSERT_TRUE(ok);
                timings[variant].push_back({std::chrono::duration<double, std::micro>(end - start).count(), cpu_us});
            }
        }
        for (int variant = 0; variant < 5; ++variant) {
            std::vector<double> wall, cpu;
            for (const auto& t : timings[variant]) {
                wall.push_back(t.wall_us);
                cpu.push_back(t.cpu_us);
            }
            std::cout << std::fixed << std::setprecision(3) << "GLM53_PREFIX_COPY blocks=" << blocks
                      << " tiles=" << batch_.tiles.size() << " bytes=" << staged_.host_bytes
                      << " sm_blocks=" << grid[variant] << " wall_median_us=" << median(wall)
                      << " thread_cpu_median_us=" << median(cpu) << std::endl;
        }
        release();
    }
}

TEST_F(Glm53PrefixCopyBenchmarkTest, BoundedWorkspaceAndCpuCost) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    unsigned original_flags = 0;
    ASSERT_EQ(cudaGetDeviceFlags(&original_flags), cudaSuccess);
    struct RestoreFlags {
        unsigned flags;
        ~RestoreFlags() {
            (void)cudaSetDeviceFlags(flags);
        }
    } restore{original_flags};
    ASSERT_EQ(cudaSetDeviceFlags((original_flags & ~cudaDeviceScheduleMask) | cudaDeviceScheduleBlockingSync),
              cudaSuccess);
    for (size_t blocks : {1u, 32u, 352u}) {
        SCOPED_TRACE(blocks);
        allocate(blocks);
        // Each candidate owns its scratch: a previous large allocation must
        // not hide the memory required by the bounded variant.
        struct Candidate {
            size_t                  cap;
            int                     grid;
            StagedMemoryCopyScratch scratch;
            std::vector<Timing>     times;
        };
        std::vector<Candidate> candidates;
        candidates.push_back({0, -1, {}, {}});
        for (size_t cap : {16u << 20, 64u << 20}) {
            for (int grid : {0, 64, 256})
                candidates.push_back({cap, grid, {}, {}});
        }
        struct ReleaseCandidates {
            std::vector<Candidate>& candidates;
            ~ReleaseCandidates() {
                for (auto& c : candidates)
                    releaseStagedMemoryCopyScratch(c.scratch);
            }
        } release_candidates{candidates};
        auto run = [&](Candidate& c) {
            return c.cap ? copyBounded(c.cap, c.grid, c.scratch) : execBatchedMemoryCopy(batch_);
        };
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto& c = candidates[i];
            for (uint8_t seed : {31, 79}) {
                fill(seed + i);
                clear();
                ASSERT_TRUE(run(c));
                verify();
            }
            ASSERT_EQ(c.scratch.host_allocation_count, 0u);
            ASSERT_LE(c.scratch.device_capacity, c.cap);
            for (int warm = 0; warm < 30; ++warm)
                ASSERT_TRUE(run(c));
        }
        for (int repeat = 0; repeat < 60; ++repeat) {
            for (size_t n = 0; n < candidates.size(); ++n) {
                auto&        c         = candidates[repeat % 2 ? n : candidates.size() - 1 - n];
                const auto   begin     = std::chrono::steady_clock::now();
                const double cpu_start = threadMicros();
                const bool   ok        = run(c);
                const double cpu       = threadMicros() - cpu_start;
                const auto   end       = std::chrono::steady_clock::now();
                ASSERT_TRUE(ok);
                c.times.push_back({std::chrono::duration<double, std::micro>(end - begin).count(), cpu});
            }
        }
        for (const auto& c : candidates) {
            std::vector<double> wall, cpu;
            for (auto t : c.times) {
                wall.push_back(t.wall_us);
                cpu.push_back(t.cpu_us);
            }
            std::cout << std::fixed << std::setprecision(3) << "GLM53_PREFIX_BOUNDED blocks=" << blocks
                      << " cap=" << c.cap << " sm_blocks=" << c.grid
                      << " payload_capacity=" << c.scratch.device_capacity
                      << " metadata_bytes=" << c.scratch.meta_capacity * (sizeof(void*) + 2 * sizeof(size_t))
                      << " wall_median_us=" << median(wall) << " thread_cpu_median_us=" << median(cpu) << std::endl;
        }
        release();
    }
}

}  // namespace
}  // namespace rtp_llm
