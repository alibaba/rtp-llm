#include <gtest/gtest.h>

#include <ATen/core/CachingHostAllocator.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"

namespace rtp_llm {
namespace test {
namespace {

// The default (1 rank x 64 MiB) is CI-safe. Reproduction runs opt in through
// RTP_LLM_PIN_STRESS_* envs and must use gpu_lock with WORLD_SIZE matching ranks.
// CUDA contexts are initialized before the allocation barrier so driver waits in
// BlockPool pin/zeroing are distinguishable from ordinary multi-process CUDA init.

using Clock = std::chrono::steady_clock;

constexpr uint64_t kMiB                    = 1024ULL * 1024ULL;
constexpr uint64_t kMaxRanks               = 8;
constexpr uint64_t kMaxPoolMiBPerRank      = 256000;
constexpr uint64_t kDefaultPoolMiBPerRank  = 64;
constexpr uint64_t kDefaultBlockMiB        = 4;
constexpr uint64_t kDefaultWriteMiB        = 16;
constexpr uint64_t kDefaultDurationSeconds = 5;
constexpr uint64_t kDefaultTimeoutSeconds  = 300;
constexpr uint64_t kDefaultCooldownSeconds = 1;
constexpr uint64_t kDefaultPinThreads      = 1;
constexpr uint64_t kDefaultPinMinKiB       = 4;
constexpr uint64_t kDefaultPinMaxKiB       = 1024;

struct StressConfig {
    uint64_t    ranks              = 1;
    uint64_t    pool_mib_per_rank  = kDefaultPoolMiBPerRank;
    uint64_t    block_mib          = kDefaultBlockMiB;
    uint64_t    write_mib          = kDefaultWriteMiB;
    uint64_t    gpu_resident_mib   = kDefaultWriteMiB;
    uint64_t    duration_seconds   = kDefaultDurationSeconds;
    uint64_t    timeout_seconds    = kDefaultTimeoutSeconds;
    uint64_t    cooldown_seconds   = kDefaultCooldownSeconds;
    uint64_t    pin_threads        = kDefaultPinThreads;
    uint64_t    driver_pin_threads = 0;
    uint64_t    pin_min_kib        = kDefaultPinMinKiB;
    uint64_t    pin_max_kib        = kDefaultPinMaxKiB;
    bool        access_pool        = true;
    std::string status_dir;
};

uint64_t readUintEnv(const char* name, uint64_t default_value, uint64_t min_value, uint64_t max_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    char*          end   = nullptr;
    const uint64_t value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || value < min_value || value > max_value) {
        std::ostringstream oss;
        oss << name << " must be an integer in [" << min_value << ", " << max_value << "], got '" << raw << "'";
        throw std::invalid_argument(oss.str());
    }
    return value;
}

StressConfig readStressConfig() {
    StressConfig config;
    config.ranks = readUintEnv("RTP_LLM_PIN_STRESS_RANKS", 1, 1, kMaxRanks);
    config.pool_mib_per_rank =
        readUintEnv("RTP_LLM_PIN_STRESS_MB_PER_RANK", kDefaultPoolMiBPerRank, 8, kMaxPoolMiBPerRank);
    config.block_mib = readUintEnv("RTP_LLM_PIN_STRESS_BLOCK_MB", kDefaultBlockMiB, 1, 1024);
    config.write_mib = readUintEnv("RTP_LLM_PIN_STRESS_WRITE_MB", kDefaultWriteMiB, 1, 1024);
    config.gpu_resident_mib =
        readUintEnv("RTP_LLM_PIN_STRESS_GPU_RESIDENT_MB", config.write_mib, config.write_mib, 262144);
    config.duration_seconds   = readUintEnv("RTP_LLM_PIN_STRESS_DURATION_SEC", kDefaultDurationSeconds, 1, 3600);
    config.timeout_seconds    = readUintEnv("RTP_LLM_PIN_STRESS_CHILD_TIMEOUT_SEC", kDefaultTimeoutSeconds, 10, 7200);
    config.cooldown_seconds   = readUintEnv("RTP_LLM_PIN_STRESS_COOLDOWN_SEC", kDefaultCooldownSeconds, 0, 600);
    config.pin_threads        = readUintEnv("RTP_LLM_PIN_STRESS_SMALL_PIN_THREADS", kDefaultPinThreads, 0, 64);
    config.driver_pin_threads = readUintEnv("RTP_LLM_PIN_STRESS_DRIVER_PIN_THREADS", 0, 0, 64);
    config.pin_min_kib        = readUintEnv("RTP_LLM_PIN_STRESS_SMALL_PIN_MIN_KB", kDefaultPinMinKiB, 1, 65536);
    config.pin_max_kib        = readUintEnv("RTP_LLM_PIN_STRESS_SMALL_PIN_MAX_KB", kDefaultPinMaxKiB, 1, 65536);
    config.access_pool        = readUintEnv("RTP_LLM_PIN_STRESS_ACCESS_POOL", 1, 0, 1) != 0;
    if (const char* status_dir = std::getenv("RTP_LLM_PIN_STRESS_STATUS_DIR")) {
        config.status_dir = status_dir;
    }

    if (config.pool_mib_per_rank % config.block_mib != 0) {
        throw std::invalid_argument("RTP_LLM_PIN_STRESS_MB_PER_RANK must be divisible by RTP_LLM_PIN_STRESS_BLOCK_MB");
    }
    if (config.access_pool && config.write_mib > config.pool_mib_per_rank) {
        throw std::invalid_argument("RTP_LLM_PIN_STRESS_WRITE_MB must not exceed the per-rank pool size");
    }
    if (config.pin_min_kib > config.pin_max_kib) {
        throw std::invalid_argument("RTP_LLM_PIN_STRESS_SMALL_PIN_MIN_KB must not exceed SMALL_PIN_MAX_KB");
    }
    return config;
}

double elapsedMs(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::string mappingForAddress(const void* ptr) {
    std::ifstream maps("/proc/self/maps");
    std::string   line;
    const auto    address = reinterpret_cast<uintptr_t>(ptr);
    while (std::getline(maps, line)) {
        uintptr_t begin = 0;
        uintptr_t end   = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2 && address >= begin && address < end) {
            return line;
        }
    }
    return "mapping_not_found";
}

void writeStatus(const StressConfig& config, int rank, const std::string& phase, const std::string& detail = "") {
    if (config.status_dir.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(config.status_dir, ec);
    std::ofstream out(config.status_dir + "/rank_" + std::to_string(rank) + ".status", std::ios::trunc);
    out << "rank=" << rank << "\n"
        << "pid=" << ::getpid() << "\n"
        << "phase=" << phase << "\n"
        << "detail=" << detail << "\n";
}

bool cudaOk(cudaError_t error, const char* operation, int rank) {
    if (error == cudaSuccess) {
        return true;
    }
    std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid() << " operation=" << operation
              << " cuda_error=" << cudaGetErrorString(error) << std::endl;
    return false;
}

void updateAtomicMax(std::atomic<uint64_t>& value, uint64_t candidate) {
    auto current = value.load(std::memory_order_relaxed);
    while (current < candidate && !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {}
}

int runChild(const StressConfig& config,
             int                 rank,
             int                 ready_fd,
             int                 allocation_barrier_fd,
             int                 pool_ready_fd,
             int                 query_barrier_fd) {
    writeStatus(config, rank, "cuda_init");
    if (!cudaOk(cudaSetDevice(rank), "cudaSetDevice", rank) || !cudaOk(cudaFree(nullptr), "cudaFree(0)", rank)) {
        return 11;
    }

    writeStatus(config, rank, "cuda_ready");
    const char ready_token = 1;
    if (::write(ready_fd, &ready_token, 1) != 1) {
        std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid() << " ready_write_failed errno=" << errno
                  << std::endl;
        return 10;
    }
    ::close(ready_fd);

    char allocation_token = 0;
    if (::read(allocation_barrier_fd, &allocation_token, 1) != 1) {
        std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid()
                  << " allocation_barrier_read_failed errno=" << errno << std::endl;
        return 10;
    }
    ::close(allocation_barrier_fd);

    const uint64_t pool_bytes  = config.pool_mib_per_rank * kMiB;
    const uint64_t block_bytes = config.block_mib * kMiB;
    const uint64_t block_num64 = pool_bytes / block_bytes;
    if (block_num64 < 2 || block_num64 > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "PIN_STRESS_ERROR rank=" << rank << " invalid_block_num=" << block_num64 << std::endl;
        return 12;
    }

    try {
        writeStatus(config, rank, "allocating", "pool_mib=" + std::to_string(config.pool_mib_per_rank));
        const auto alloc_begin = Clock::now();
        auto       pool_config = BlockPoolConfigHelper::createConfig(
            1, static_cast<uint32_t>(block_num64), static_cast<size_t>(block_bytes), rtp_llm::TYPE_INT8);
        auto pool = std::make_unique<BlockPool>(pool_config, AllocationType::HOST);
        if (!pool->init()) {
            std::cerr << "PIN_STRESS_ERROR rank=" << rank << " BlockPool::init returned false" << std::endl;
            return 13;
        }
        const double alloc_ms = elapsedMs(alloc_begin);
        if (pool->where() != MemoryType::MEMORY_CPU_PINNED) {
            std::cerr << "PIN_STRESS_ERROR rank=" << rank
                      << " expected pinned host backing, actual=" << static_cast<int>(pool->where()) << std::endl;
            return 14;
        }

        const std::string mapping = mappingForAddress(pool->getBaseAddress());
        if (mapping.find("/dev/shm") != std::string::npos || mapping.find("/run/shm") != std::string::npos) {
            std::cerr << "PIN_STRESS_ERROR rank=" << rank << " shared_memory_filesystem_mapping=" << mapping
                      << std::endl;
            return 15;
        }
        std::cout << "PIN_STRESS_ALLOC rank=" << rank << " pid=" << ::getpid() << " device=" << rank
                  << " pool_mib=" << config.pool_mib_per_rank << " alloc_ms=" << alloc_ms << " mapping='" << mapping
                  << "'" << std::endl;

        writeStatus(config, rank, "pool_ready", "waiting_for_all_ranks");
        if (::write(pool_ready_fd, &ready_token, 1) != 1) {
            std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid()
                      << " pool_ready_write_failed errno=" << errno << std::endl;
            return 20;
        }
        ::close(pool_ready_fd);

        char query_token = 0;
        if (::read(query_barrier_fd, &query_token, 1) != 1) {
            std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid()
                      << " query_barrier_read_failed errno=" << errno << std::endl;
            return 20;
        }
        ::close(query_barrier_fd);

        const size_t write_bytes        = static_cast<size_t>(config.write_mib * kMiB);
        const size_t gpu_resident_bytes = static_cast<size_t>(config.gpu_resident_mib * kMiB);
        void*        device_buffer      = nullptr;
        cudaStream_t stream             = nullptr;
        if (!cudaOk(cudaMalloc(&device_buffer, gpu_resident_bytes), "cudaMalloc", rank)
            || !cudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags", rank)
            || !cudaOk(cudaMemsetAsync(device_buffer, 0x40 + rank, gpu_resident_bytes, stream),
                       "cudaMemsetAsync(resident)",
                       rank)
            || !cudaOk(cudaStreamSynchronize(stream), "cudaStreamSynchronize(init)", rank)) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
            if (device_buffer != nullptr) {
                cudaFree(device_buffer);
            }
            return 16;
        }

        writeStatus(config,
                    rank,
                    "query_stress",
                    "write_mib=" + std::to_string(config.write_mib) + " gpu_resident_mib="
                        + std::to_string(config.gpu_resident_mib) + " access_pool=" + std::to_string(config.access_pool)
                        + " small_pin_threads=" + std::to_string(config.pin_threads)
                        + " driver_pin_threads=" + std::to_string(config.driver_pin_threads));
        const auto    stress_begin      = Clock::now();
        const auto    deadline          = stress_begin + std::chrono::seconds(config.duration_seconds);
        const size_t  pool_chunk_count  = std::max<size_t>(1, static_cast<size_t>(pool_bytes) / write_bytes);
        const size_t  gpu_chunk_count   = std::max<size_t>(1, gpu_resident_bytes / write_bytes);
        size_t        iterations        = 0;
        double        max_copy_ms       = 0.0;
        uint64_t      bytes_written     = 0;
        auto*         host_base         = static_cast<uint8_t*>(pool->getBaseAddress());
        const uint8_t expected          = static_cast<uint8_t>(0x40 + rank);
        auto*         host_allocator    = at::getHostAllocator(at::kCUDA);
        const auto    host_stats_before = host_allocator->get_stats();

        std::atomic<bool>        stop_pin_workers{false};
        std::atomic<bool>        pin_failed{false};
        std::atomic<uint64_t>    pin_iterations{0};
        std::atomic<uint64_t>    pin_bytes{0};
        std::atomic<uint64_t>    pin_total_us{0};
        std::atomic<uint64_t>    pin_max_us{0};
        std::vector<std::thread> pin_workers;
        pin_workers.reserve(config.pin_threads);
        for (uint64_t worker = 0; worker < config.pin_threads; ++worker) {
            pin_workers.emplace_back([&, worker]() {
                uint64_t size_kib = config.pin_min_kib;
                for (uint64_t step = 0; step < worker && size_kib < config.pin_max_kib; ++step) {
                    size_kib = size_kib > config.pin_max_kib / 2 ? config.pin_max_kib : size_kib * 2;
                }
                try {
                    while (!stop_pin_workers.load(std::memory_order_relaxed)) {
                        const auto begin = Clock::now();
                        {
                            // Match the application-level query path: short-lived pinned
                            // tensors are created through PyTorch's caching host allocator.
                            auto tensor = torch::empty(
                                {static_cast<int64_t>(size_kib * 1024)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
                            auto* data               = tensor.data_ptr<uint8_t>();
                            data[0]                  = static_cast<uint8_t>(rank);
                            data[tensor.numel() - 1] = static_cast<uint8_t>(worker);
                        }
                        const auto us = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count());
                        pin_iterations.fetch_add(1, std::memory_order_relaxed);
                        pin_bytes.fetch_add(size_kib * 1024, std::memory_order_relaxed);
                        pin_total_us.fetch_add(us, std::memory_order_relaxed);
                        updateAtomicMax(pin_max_us, us);
                        size_kib = size_kib >= config.pin_max_kib ? config.pin_min_kib :
                                                                    std::min(config.pin_max_kib, size_kib * 2);
                    }
                } catch (const std::exception& error) {
                    pin_failed.store(true, std::memory_order_relaxed);
                    stop_pin_workers.store(true, std::memory_order_relaxed);
                    std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pin_worker=" << worker << " exception='"
                              << error.what() << "'" << std::endl;
                }
            });
        }

        std::atomic<uint64_t>    driver_pin_iterations{0};
        std::atomic<uint64_t>    driver_pin_bytes{0};
        std::atomic<uint64_t>    driver_pin_total_us{0};
        std::atomic<uint64_t>    driver_pin_max_us{0};
        std::vector<std::thread> driver_pin_workers;
        driver_pin_workers.reserve(config.driver_pin_threads);
        for (uint64_t worker = 0; worker < config.driver_pin_threads; ++worker) {
            driver_pin_workers.emplace_back([&, worker]() {
                if (!cudaOk(cudaSetDevice(rank), "cudaSetDevice(driver_pin_worker)", rank)) {
                    pin_failed.store(true, std::memory_order_relaxed);
                    stop_pin_workers.store(true, std::memory_order_relaxed);
                    return;
                }
                uint64_t size_kib = config.pin_min_kib;
                for (uint64_t step = 0; step < worker && size_kib < config.pin_max_kib; ++step) {
                    size_kib = size_kib > config.pin_max_kib / 2 ? config.pin_max_kib : size_kib * 2;
                }
                while (!stop_pin_workers.load(std::memory_order_relaxed)) {
                    const auto  begin = Clock::now();
                    void*       ptr   = nullptr;
                    cudaError_t error = cudaHostAlloc(&ptr, static_cast<size_t>(size_kib * 1024), cudaHostAllocDefault);
                    if (error == cudaSuccess) {
                        auto* data                = static_cast<uint8_t*>(ptr);
                        data[0]                   = static_cast<uint8_t>(rank);
                        data[size_kib * 1024 - 1] = static_cast<uint8_t>(worker);
                        error                     = cudaFreeHost(ptr);
                    }
                    if (!cudaOk(error, "cudaHostAlloc/cudaFreeHost(driver_pin_worker)", rank)) {
                        pin_failed.store(true, std::memory_order_relaxed);
                        stop_pin_workers.store(true, std::memory_order_relaxed);
                        return;
                    }
                    const auto us = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count());
                    driver_pin_iterations.fetch_add(1, std::memory_order_relaxed);
                    driver_pin_bytes.fetch_add(size_kib * 1024, std::memory_order_relaxed);
                    driver_pin_total_us.fetch_add(us, std::memory_order_relaxed);
                    updateAtomicMax(driver_pin_max_us, us);
                    size_kib = size_kib >= config.pin_max_kib ? config.pin_min_kib :
                                                                std::min(config.pin_max_kib, size_kib * 2);
                }
            });
        }

        bool copy_failed = false;
        do {
            const auto copy_begin = Clock::now();
            if (config.access_pool) {
                // A prime stride spreads writes across the backing instead of repeatedly hitting its prefix.
                const size_t chunk = (iterations * 104729ULL + static_cast<size_t>(rank) * 8191ULL) % pool_chunk_count;
                auto*        dst   = host_base + chunk * write_bytes;
                if (!cudaOk(cudaMemcpyAsync(dst, device_buffer, write_bytes, cudaMemcpyDeviceToHost, stream),
                            "cudaMemcpyAsync(D2H)",
                            rank)
                    || !cudaOk(cudaStreamSynchronize(stream), "cudaStreamSynchronize(copy)", rank)) {
                    copy_failed = true;
                    break;
                }
                if (dst[0] != expected || dst[write_bytes - 1] != expected) {
                    std::cerr << "PIN_STRESS_ERROR rank=" << rank << " verification_failed iteration=" << iterations
                              << " chunk=" << chunk << std::endl;
                    copy_failed = true;
                    break;
                }
            } else {
                const size_t chunk   = (iterations * 104729ULL + static_cast<size_t>(rank) * 8191ULL) % gpu_chunk_count;
                auto*        gpu_dst = static_cast<uint8_t*>(device_buffer) + chunk * write_bytes;
                if (!cudaOk(cudaMemsetAsync(gpu_dst, 0x40 + rank, write_bytes, stream),
                            "cudaMemsetAsync(query_gpu_only)",
                            rank)
                    || !cudaOk(cudaStreamSynchronize(stream), "cudaStreamSynchronize(query_gpu_only)", rank)) {
                    copy_failed = true;
                    break;
                }
            }
            max_copy_ms = std::max(max_copy_ms, elapsedMs(copy_begin));
            ++iterations;
            bytes_written += write_bytes;
        } while (Clock::now() < deadline);

        stop_pin_workers.store(true, std::memory_order_relaxed);
        for (auto& worker : pin_workers) {
            worker.join();
        }
        for (auto& worker : driver_pin_workers) {
            worker.join();
        }
        const auto host_stats_after = host_allocator->get_stats();

        cudaStreamDestroy(stream);
        cudaFree(device_buffer);
        if (copy_failed || pin_failed.load(std::memory_order_relaxed)) {
            return 17;
        }

        writeStatus(config, rank, "teardown");
        const auto teardown_begin = Clock::now();
        pool.reset();
        const double teardown_ms = elapsedMs(teardown_begin);
        writeStatus(config, rank, "done");
        std::cout << "PIN_STRESS_RESULT rank=" << rank << " pid=" << ::getpid()
                  << " pool_mib=" << config.pool_mib_per_rank << " iterations=" << iterations
                  << " access_pool=" << config.access_pool << " gpu_resident_mib=" << config.gpu_resident_mib
                  << " bytes_written=" << bytes_written << " stress_ms=" << elapsedMs(stress_begin)
                  << " max_copy_ms=" << max_copy_ms << " small_pin_iterations=" << pin_iterations.load()
                  << " small_pin_bytes=" << pin_bytes.load() << " small_pin_total_us=" << pin_total_us.load()
                  << " small_pin_max_us=" << pin_max_us.load()
                  << " driver_pin_iterations=" << driver_pin_iterations.load()
                  << " driver_pin_bytes=" << driver_pin_bytes.load()
                  << " driver_pin_total_us=" << driver_pin_total_us.load()
                  << " driver_pin_max_us=" << driver_pin_max_us.load() << " host_allocator_alloc_calls="
                  << host_stats_after.host_alloc_time.count - host_stats_before.host_alloc_time.count
                  << " host_allocator_alloc_us="
                  << host_stats_after.host_alloc_time.total - host_stats_before.host_alloc_time.total
                  << " host_allocator_free_calls="
                  << host_stats_after.host_free_time.count - host_stats_before.host_free_time.count
                  << " host_allocator_free_us="
                  << host_stats_after.host_free_time.total - host_stats_before.host_free_time.total
                  << " teardown_ms=" << teardown_ms << std::endl;
        return 0;
    } catch (const std::exception& error) {
        writeStatus(config, rank, "exception", error.what());
        std::cerr << "PIN_STRESS_ERROR rank=" << rank << " pid=" << ::getpid() << " exception='" << error.what() << "'"
                  << std::endl;
        return 19;
    }
}

}  // namespace

TEST(PinnedHostKVWriteStressTest, InvalidPinModeFailsFast) {
    ASSERT_EQ(::setenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "invalid", 1), 0);

    auto pool_config = BlockPoolConfigHelper::createConfig(1, 2, 4 * kMiB, rtp_llm::TYPE_INT8);
    try {
        BlockPool pool(pool_config, AllocationType::HOST);
        EXPECT_THROW(pool.init(), std::invalid_argument);
    } catch (...) {
        (void)::unsetenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE");
        throw;
    }

    ASSERT_EQ(::unsetenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE"), 0);
}

TEST(PinnedHostKVWriteStressTest, MultiRankPinnedAllocationWriteAndTeardown) {
    const StressConfig config  = readStressConfig();
    const char*        pin_env = std::getenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    ASSERT_NE(pin_env, nullptr);
    ASSERT_STREQ(pin_env, "1") << "This test only exercises pinned host BlockPool backing";

    std::cout << "PIN_STRESS_CONFIG parent_pid=" << ::getpid() << " ranks=" << config.ranks
              << " pool_mib_per_rank=" << config.pool_mib_per_rank
              << " aggregate_mib=" << config.ranks * config.pool_mib_per_rank << " block_mib=" << config.block_mib
              << " write_mib=" << config.write_mib << " gpu_resident_mib=" << config.gpu_resident_mib
              << " access_pool=" << config.access_pool << " duration_sec=" << config.duration_seconds
              << " cooldown_sec=" << config.cooldown_seconds << " small_pin_threads=" << config.pin_threads
              << " driver_pin_threads=" << config.driver_pin_threads << " small_pin_kib_range=" << config.pin_min_kib
              << "-" << config.pin_max_kib << " timeout_sec=" << config.timeout_seconds << " status_dir='"
              << config.status_dir << "'" << std::endl;

    int ready_pipe[2]         = {-1, -1};
    int allocation_barrier[2] = {-1, -1};
    int pool_ready_pipe[2]    = {-1, -1};
    int query_barrier[2]      = {-1, -1};
    ASSERT_EQ(::pipe(ready_pipe), 0) << "ready pipe failed: " << std::strerror(errno);
    ASSERT_EQ(::pipe(allocation_barrier), 0) << "allocation barrier pipe failed: " << std::strerror(errno);
    ASSERT_EQ(::pipe(pool_ready_pipe), 0) << "pool ready pipe failed: " << std::strerror(errno);
    ASSERT_EQ(::pipe(query_barrier), 0) << "query barrier pipe failed: " << std::strerror(errno);

    std::vector<pid_t> children;
    children.reserve(config.ranks);
    for (uint64_t rank = 0; rank < config.ranks; ++rank) {
        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0) << "fork failed at rank " << rank << ": " << std::strerror(errno);
        if (pid == 0) {
            ::close(ready_pipe[0]);
            ::close(allocation_barrier[1]);
            ::close(pool_ready_pipe[0]);
            ::close(query_barrier[1]);
            const int exit_code = runChild(config,
                                           static_cast<int>(rank),
                                           ready_pipe[1],
                                           allocation_barrier[0],
                                           pool_ready_pipe[1],
                                           query_barrier[0]);
            std::cout.flush();
            std::cerr.flush();
            ::_exit(exit_code);
        }
        children.push_back(pid);
        std::cout << "PIN_STRESS_CHILD rank=" << rank << " pid=" << pid << std::endl;
    }
    ::close(ready_pipe[1]);
    ::close(allocation_barrier[0]);
    ::close(pool_ready_pipe[1]);
    ::close(query_barrier[0]);

    std::vector<char> ready_tokens(config.ranks, 0);
    size_t            ready_count = 0;
    while (ready_count < ready_tokens.size()) {
        const ssize_t count =
            ::read(ready_pipe[0], ready_tokens.data() + ready_count, ready_tokens.size() - ready_count);
        ASSERT_GT(count, 0) << "ready pipe closed before every CUDA context initialized: " << std::strerror(errno);
        ready_count += static_cast<size_t>(count);
    }
    ::close(ready_pipe[0]);
    std::cout << "PIN_STRESS_ALL_CUDA_READY ranks=" << ready_count << std::endl;

    std::vector<char> allocation_tokens(config.ranks, 1);
    ASSERT_EQ(::write(allocation_barrier[1], allocation_tokens.data(), allocation_tokens.size()),
              static_cast<ssize_t>(allocation_tokens.size()))
        << "allocation barrier write failed: " << std::strerror(errno);
    ::close(allocation_barrier[1]);

    std::vector<char> pool_ready_tokens(config.ranks, 0);
    size_t            pool_ready_count = 0;
    while (pool_ready_count < pool_ready_tokens.size()) {
        const ssize_t count = ::read(pool_ready_pipe[0],
                                     pool_ready_tokens.data() + pool_ready_count,
                                     pool_ready_tokens.size() - pool_ready_count);
        ASSERT_GT(count, 0) << "pool ready pipe closed before every pool initialized: " << std::strerror(errno);
        pool_ready_count += static_cast<size_t>(count);
    }
    ::close(pool_ready_pipe[0]);
    std::cout << "PIN_STRESS_ALL_POOLS_READY ranks=" << pool_ready_count << " cooldown_sec=" << config.cooldown_seconds
              << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(config.cooldown_seconds));

    std::vector<char> query_tokens(config.ranks, 1);
    ASSERT_EQ(::write(query_barrier[1], query_tokens.data(), query_tokens.size()),
              static_cast<ssize_t>(query_tokens.size()))
        << "query barrier write failed: " << std::strerror(errno);
    ::close(query_barrier[1]);
    std::cout << "PIN_STRESS_QUERY_START ranks=" << config.ranks << std::endl;

    std::vector<bool> completed(children.size(), false);
    size_t            remaining = children.size();
    bool              success   = true;
    const auto        deadline  = Clock::now() + std::chrono::seconds(config.timeout_seconds);
    while (remaining > 0 && Clock::now() < deadline) {
        for (size_t rank = 0; rank < children.size(); ++rank) {
            if (completed[rank]) {
                continue;
            }
            int         status = 0;
            const pid_t waited = ::waitpid(children[rank], &status, WNOHANG);
            if (waited == 0) {
                continue;
            }
            ASSERT_EQ(waited, children[rank]) << "waitpid failed for rank " << rank << ": " << std::strerror(errno);
            completed[rank] = true;
            --remaining;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                success = false;
                std::cerr << "PIN_STRESS_CHILD_FAILURE rank=" << rank << " pid=" << children[rank]
                          << " status=" << status << std::endl;
            }
        }
        if (remaining > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    if (remaining > 0) {
        success = false;
        for (size_t rank = 0; rank < children.size(); ++rank) {
            if (!completed[rank]) {
                std::cerr << "PIN_STRESS_TIMEOUT rank=" << rank << " pid=" << children[rank]
                          << " timeout_sec=" << config.timeout_seconds << std::endl;
                ::kill(children[rank], SIGTERM);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
        for (size_t rank = 0; rank < children.size(); ++rank) {
            if (!completed[rank]) {
                int status = 0;
                if (::waitpid(children[rank], &status, WNOHANG) == 0) {
                    ::kill(children[rank], SIGKILL);
                }
                ::waitpid(children[rank], &status, 0);
            }
        }
    }

    EXPECT_TRUE(success) << "one or more pinned-memory stress ranks failed; inspect PIN_STRESS_* output";
}

}  // namespace test
}  // namespace rtp_llm
