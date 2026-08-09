#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkRunner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkWorkload.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::benchmark {

namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsedNs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

bool touchesDevice(const std::string& direction) {
    return direction == "d2h" || direction == "h2d" || direction == "d2disk" || direction == "disk2d";
}

bool touchesHost(const std::string& direction) {
    return direction == "d2h" || direction == "h2d" || direction == "h2disk" || direction == "disk2h";
}

bool touchesDisk(const std::string& direction) {
    return direction == "d2disk" || direction == "disk2d" || direction == "h2disk" || direction == "disk2h";
}

bool readsDisk(const std::string& direction) {
    return direction == "disk2d" || direction == "disk2h";
}

bool writesDisk(const std::string& direction) {
    return direction == "d2disk" || direction == "h2disk";
}

class RecordingDeviceHostCopyStrategy final: public DeviceHostCopyStrategy {
public:
    RecordingDeviceHostCopyStrategy(std::unique_ptr<DeviceHostCopyStrategy> delegate, std::atomic<size_t>* completed):
        delegate_(std::move(delegate)), completed_(completed) {}

    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override {
        auto result = delegate_->tryExecute(plan, options);
        if (result.status == StrategyStatus::DONE) {
            completed_->fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

private:
    std::unique_ptr<DeviceHostCopyStrategy> delegate_;
    std::atomic<size_t>*                    completed_;
};

void installStrategyRecorders(PerRankBlockTransferEngine&                          engine,
                              const std::shared_ptr<BenchmarkDeviceHostCopyStats>& stats) {
    auto& executor   = *engine.device_host_executor_;
    auto& strategies = executor.strategies_;
    RTP_LLM_CHECK(strategies.size() == 3);
    for (auto& strategy : strategies) {
        std::atomic<size_t>* completed = nullptr;
        if (dynamic_cast<StagedSmDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->staged_sm;
        } else if (dynamic_cast<CudaBatchDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->cuda_batch;
        } else if (dynamic_cast<GenericMultiCopyDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->generic;
        }
        RTP_LLM_CHECK(completed != nullptr);
        strategy = std::make_unique<RecordingDeviceHostCopyStrategy>(std::move(strategy), completed);
    }
}

}  // namespace

size_t TransferBenchmarkRunner::BatchResult::attempted() const {
    return std::accumulate(directions.begin(), directions.end(), size_t{0}, [](size_t total, const auto& entry) {
        return total + entry.second.attempted;
    });
}

size_t TransferBenchmarkRunner::BatchResult::succeeded() const {
    return std::accumulate(directions.begin(), directions.end(), size_t{0}, [](size_t total, const auto& entry) {
        return total + entry.second.succeeded;
    });
}

size_t TransferBenchmarkRunner::BatchResult::failed() const {
    return std::accumulate(directions.begin(), directions.end(), size_t{0}, [](size_t total, const auto& entry) {
        return total + entry.second.failed;
    });
}

size_t TransferBenchmarkRunner::BatchResult::visitedWorkingSetBlocks() const {
    return static_cast<size_t>(std::count(visited_working_set.begin(), visited_working_set.end(), true));
}

TransferBenchmarkRunner::TransferBenchmarkRunner(const ModelProfile&    profile,
                                                 const TransferOptions& options,
                                                 uint64_t               seed,
                                                 const std::string&     output_json_path):
    profile_(profile), options_(options), seed_(seed), output_json_path_(output_json_path) {
    writer_.setRunner("transfer");
    writer_.setModelProfile(profile_.profile_id, profile_.sha256_hex);
}

bool TransferBenchmarkRunner::run() {
    std::string directions;
    for (size_t i = 0; i < options_.transfer_directions.size(); ++i) {
        directions += (i == 0 ? "" : "+") + options_.transfer_directions[i];
    }
    writer_.setMeasurement(directions + "_" + options_.group_set);
    const bool benchmark_ok = runPurePathTransfer();
    writer_.setStatus(benchmark_ok ? "completed" : "failed");

    bool output_ok = true;
    if (!output_json_path_.empty()) {
        std::ofstream output(output_json_path_, std::ios::trunc);
        output_ok = output.is_open();
        if (output_ok) {
            output << writer_.toJson() << '\n';
            output_ok = output.good();
        }
        if (!output_ok) {
            std::cerr << "Failed to write result JSON: " << output_json_path_ << std::endl;
        }
    }
    return benchmark_ok && output_ok;
}

void TransferBenchmarkRunner::recordBatchFailure(const std::string& phase, const BatchResult& batch) {
    writer_.addResolvedConfigInt(phase + ".attempted", batch.attempted());
    writer_.addResolvedConfigInt(phase + ".failed", batch.failed());
    for (const auto& [direction, stats] : batch.directions) {
        if (!stats.first_error.empty()) {
            writer_.addResolvedConfig(phase + "." + direction + ".first_error", stats.first_error);
            writer_.addResolvedConfig(phase + "." + direction + ".first_failure_type", stats.first_failure_type);
        }
    }
}

TransferBenchmarkRunner::TransferSetup
TransferBenchmarkRunner::buildTransferSetup(const GroupSetInfo&            gs_info,
                                            size_t                         device_block_count,
                                            const std::string&             pool_prefix,
                                            bool                           need_device,
                                            std::shared_ptr<HostBlockPool> host_pool,
                                            BlockTreeDiskBlockPoolPtr      disk_pool) {
    TransferSetup setup;
    for (const auto& tag : gs_info.member_tags) {
        const auto* group = profile_.findGroup(tag);
        RTP_LLM_CHECK_WITH_INFO(
            group != nullptr, "group set %s references unknown group %s", gs_info.name.c_str(), tag.c_str());
        setup.members.push_back(group);
    }
    RTP_LLM_CHECK(!setup.members.empty());

    std::vector<std::pair<std::string, rtp_llm::CacheGroupType>> group_specs;
    std::vector<size_t>                                          layer_strides;
    std::vector<size_t>                                          layer_counts;
    std::vector<size_t>                                          group_ids;
    size_t                                                       tile_count = 0;
    bool                                                         is_swa     = false;
    for (size_t member_index = 0; member_index < setup.members.size(); ++member_index) {
        const auto* member = setup.members[member_index];
        if (need_device) {
            setup.device_pools.push_back(BenchmarkFixture::createDevicePool(
                member->layer_stride_bytes, member->layer_count, device_block_count, pool_prefix + member->tag));
        }
        const auto type =
            member->type == CacheGroupType::SWA ? rtp_llm::CacheGroupType::SWA : rtp_llm::CacheGroupType::FULL;
        group_specs.emplace_back(member->tag, type);
        layer_strides.push_back(member->layer_stride_bytes);
        layer_counts.push_back(member->layer_count);
        group_ids.push_back(member_index);
        tile_count += member->layer_count;
        is_swa |= member->type == CacheGroupType::SWA;
    }

    auto topology = BenchmarkFixture::createTopology(group_specs, layer_strides, layer_counts);
    if (is_swa) {
        setup.group_set =
            BenchmarkFixture::createSWAGroupSet(setup.device_pools, host_pool, disk_pool, 0, topology, group_ids, 128);
    } else {
        setup.group_set =
            BenchmarkFixture::createFullGroupSet(setup.device_pools, host_pool, disk_pool, 0, topology, group_ids);
    }

    setup.copy_stats = std::make_shared<BenchmarkDeviceHostCopyStats>();
    DeviceHostCopyOptions copy_options;
    if (options_.copy_strategy == "staged-sm") {
        copy_options.staged_sm_copy_enabled   = true;
        copy_options.staged_sm_min_tile_count = 0;
        copy_options.staged_sm_min_bytes      = 0;
        copy_options.cuda_batch_copy_enabled  = false;
    } else if (options_.copy_strategy == "batch") {
        copy_options.staged_sm_copy_enabled  = false;
        copy_options.cuda_batch_copy_enabled = true;
    }
    // Staging leases bound in-flight device<->disk ops; with more workers than
    // leases the pool is guaranteed to be exhausted, so floor it at concurrency.
    const size_t staging_count = std::max(options_.device_disk_staging_block_count, device_block_count);
    setup.engine               = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{setup.group_set}, copy_options, staging_count);
    installStrategyRecorders(*setup.engine, setup.copy_stats);

    writer_.addResolvedConfig("requested_copy_strategy", options_.copy_strategy);
    writer_.addResolvedConfigInt("member_group_count", setup.members.size());
    writer_.addResolvedConfigInt("copy_tile_count", tile_count);
    return setup;
}

bool TransferBenchmarkRunner::runPurePathTransfer() {
    const auto* group_set_info = profile_.findGroupSet(options_.group_set);
    if (group_set_info == nullptr) {
        throw std::runtime_error("Unknown group set: " + options_.group_set);
    }

    static const std::vector<std::string> kValidDirections = {"d2h", "h2d", "d2disk", "disk2d", "h2disk", "disk2h"};
    for (const auto& direction : options_.transfer_directions) {
        if (std::find(kValidDirections.begin(), kValidDirections.end(), direction) == kValidDirections.end()) {
            throw std::runtime_error("Unknown transfer direction: " + direction);
        }
    }

    const bool need_device =
        std::any_of(options_.transfer_directions.begin(), options_.transfer_directions.end(), touchesDevice);
    const bool need_host =
        std::any_of(options_.transfer_directions.begin(), options_.transfer_directions.end(), touchesHost);
    const bool need_disk =
        std::any_of(options_.transfer_directions.begin(), options_.transfer_directions.end(), touchesDisk);
    if (need_disk && options_.disk_path.empty()) {
        throw std::runtime_error("Disk directions require --disk-path");
    }
    if (!need_device && options_.copy_strategy != "auto") {
        throw std::runtime_error("--copy-strategy is only valid for a transfer path that touches device memory");
    }

    const size_t payload_bytes       = group_set_info->payload_bytes;
    const size_t concurrency         = options_.transfer_concurrency;
    const size_t working_set_blocks  = options_.working_set_blocks == 0 ? concurrency * 4 : options_.working_set_blocks;
    const bool   host_is_working_set = !need_disk;
    const size_t host_block_count =
        need_host ? (host_is_working_set ? working_set_blocks : concurrency) : (need_disk ? 1 : 0);

    writer_.setPayloadMode("model_sized", payload_bytes);
    writer_.addResolvedConfigInt("requested_operation_count", options_.transfer_operation_count);
    writer_.addResolvedConfigInt("requested_working_set_blocks", working_set_blocks);
    writer_.addResolvedConfigInt("transfer_concurrency", concurrency);
    writer_.addResolvedConfig("disk_io_mode", options_.disk_io_mode);
    writer_.addResolvedConfig("disk_access_pattern", options_.disk_access_pattern);

    std::shared_ptr<HostBlockPool> host_pool;
    BlockTreeDiskBlockPoolPtr      disk_pool;
    if (host_block_count > 0) {
        host_pool = BenchmarkFixture::createHostPool(payload_bytes, host_block_count, options_.host_memory == "pinned");
    }
    if (need_disk) {
        disk_pool = BenchmarkFixture::createDiskPool(payload_bytes,
                                                     working_set_blocks,
                                                     createDiskWorkDir(),
                                                     "transfer_disk",
                                                     options_.disk_io_mode == "buffered");
    }

    auto setup = buildTransferSetup(*group_set_info, concurrency, "transfer_", need_device, host_pool, disk_pool);

    MemberDeviceBlocks device_blocks(setup.device_pools.size());
    for (size_t member_index = 0; member_index < setup.device_pools.size(); ++member_index) {
        for (size_t worker = 0; worker < concurrency; ++worker) {
            device_blocks[member_index].push_back(setup.device_pools[member_index]->malloc().value());
        }
    }
    std::vector<BlockIdxType> host_blocks;
    for (size_t block_index = 0; block_index < host_block_count; ++block_index) {
        const BlockIdxType block = host_pool->malloc().value();
        host_blocks.push_back(block);
        const auto buffer = host_pool->blockBuffer(block);
        std::memset(buffer.addr, 0, buffer.stride_bytes);
    }
    std::vector<BlockIdxType> disk_blocks;
    for (size_t block_index = 0; block_index < working_set_blocks && need_disk; ++block_index) {
        disk_blocks.push_back(disk_pool->malloc().value());
    }
    for (size_t member_index = 0; member_index < setup.device_pools.size(); ++member_index) {
        for (const BlockIdxType block : device_blocks[member_index]) {
            for (size_t layer = 0; layer < setup.members[member_index]->layer_count; ++layer) {
                for (const auto& buffer : setup.device_pools[member_index]->convertIndexToBuffer(layer, block)) {
                    cudaMemset(buffer.addr, 0, buffer.size_bytes);
                }
            }
        }
    }
    cudaDeviceSynchronize();

    bool disk_initialized = false;
    bool disk_prefill     = false;
    for (const auto& direction : options_.transfer_directions) {
        if (readsDisk(direction) && !disk_initialized) {
            disk_prefill = true;
        }
        disk_initialized |= writesDisk(direction);
    }
    if (disk_prefill) {
        const auto source = host_pool->blockBuffer(host_blocks.front());
        for (const BlockIdxType block : disk_blocks) {
            if (disk_pool->write(block, source.addr, disk_pool->strideBytes()) != BlockIOStatus::OK) {
                throw std::runtime_error("Failed to prefill disk working set");
            }
        }
        writer_.addResolvedConfig("read_precondition", "disk_working_set_prefilled");
    } else if (need_disk) {
        writer_.addResolvedConfig("read_precondition", "paired_write_same_coordinate");
    } else {
        writer_.addResolvedConfig("read_precondition", "initialized_worker_and_host_blocks");
    }

    writer_.addResolvedConfigInt("tier.device.capacity_blocks", need_device ? concurrency : 0);
    writer_.addResolvedConfigInt("tier.device.allocated_blocks", need_device ? concurrency : 0);
    writer_.addResolvedConfigInt("tier.device.addressable_blocks", need_device ? concurrency : 0);
    writer_.addResolvedConfigInt("tier.host.capacity_blocks", host_block_count);
    writer_.addResolvedConfigInt("tier.host.allocated_blocks", host_blocks.size());
    writer_.addResolvedConfigInt("tier.host.addressable_blocks",
                                 need_host ? (host_is_working_set ? working_set_blocks : concurrency) : 0);
    writer_.addResolvedConfigInt("tier.disk.capacity_blocks", need_disk ? working_set_blocks : 0);
    writer_.addResolvedConfigInt("tier.disk.allocated_blocks", disk_blocks.size());
    writer_.addResolvedConfigInt("tier.disk.addressable_blocks", need_disk ? working_set_blocks : 0);

    size_t        coordinate_cursor = 0;
    const size_t  direction_count   = options_.transfer_directions.size();
    const size_t  warmup_operations = working_set_blocks * direction_count;
    const auto    warmup_start      = Clock::now();
    const auto    warmup            = runTransferBatch(setup.engine,
                                         options_.transfer_directions,
                                         device_blocks,
                                         host_blocks,
                                         disk_blocks,
                                         concurrency,
                                         warmup_operations,
                                         coordinate_cursor,
                                         working_set_blocks,
                                         host_is_working_set);
    const int64_t warmup_ns         = elapsedNs(warmup_start, Clock::now());
    if (warmup.failed() != 0 || warmup.attempted() != warmup_operations) {
        recordBatchFailure("warmup", warmup);
        return false;
    }
    coordinate_cursor += working_set_blocks;
    writer_.addPhaseNs("warmup", warmup_ns);
    writer_.addResolvedConfig("cache_precondition",
                              options_.disk_io_mode == "buffered" ? "full_working_set_sweep" : "direct_or_memory");

    const size_t min_seconds           = options_.min_measured_seconds;
    size_t       pilot_operations      = std::max<size_t>(64, options_.transfer_operation_count / 32);
    int64_t      pilot_ns              = 0;
    size_t       calibrated_operations = options_.transfer_operation_count;
    for (int attempt = 0; attempt < 4; ++attempt) {
        const auto pilot_start = Clock::now();
        const auto pilot       = runTransferBatch(setup.engine,
                                            options_.transfer_directions,
                                            device_blocks,
                                            host_blocks,
                                            disk_blocks,
                                            concurrency,
                                            pilot_operations,
                                            coordinate_cursor,
                                            working_set_blocks,
                                            host_is_working_set);
        pilot_ns               = elapsedNs(pilot_start, Clock::now());
        if (pilot.failed() != 0 || pilot.attempted() != pilot_operations) {
            recordBatchFailure("pilot", pilot);
            return false;
        }
        coordinate_cursor += (pilot_operations + direction_count - 1) / direction_count;
        if (pilot_ns >= 300'000'000 || attempt == 3) {
            if (pilot_ns > 0) {
                const double scale    = static_cast<double>(min_seconds) * 1.05e9 / static_cast<double>(pilot_ns);
                calibrated_operations = static_cast<size_t>(static_cast<double>(pilot_operations) * scale);
            }
            break;
        }
        pilot_operations *= 2;
    }

    const size_t initial_measured_operations =
        std::max({options_.transfer_operation_count, calibrated_operations, working_set_blocks * direction_count});
    std::cout << "[transfer] pilot " << pilot_operations << " global ops in " << static_cast<double>(pilot_ns) / 1e6
              << " ms; initial measured global ops=" << initial_measured_operations << std::endl;

    setup.copy_stats->reset();
    std::cout << "MEASURE_START" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    const auto start            = Clock::now();
    auto       measured         = runTransferBatch(setup.engine,
                                     options_.transfer_directions,
                                     device_blocks,
                                     host_blocks,
                                     disk_blocks,
                                     concurrency,
                                     initial_measured_operations,
                                     coordinate_cursor,
                                     working_set_blocks,
                                     host_is_working_set);
    size_t     final_operations = initial_measured_operations;
    coordinate_cursor += (initial_measured_operations + direction_count - 1) / direction_count;

    // Pilot calibration is only an estimate. Buffered IO in particular may
    // become faster after the pilot as the page cache and writeback pipeline
    // warm up, so a fixed 5% calibration margin does not guarantee the
    // documented duration floor. Extend the same measured window until the
    // wall-clock floor is actually reached, merging all counters and visited
    // working-set bits into the final result.
    const int64_t min_measured_ns = static_cast<int64_t>(min_seconds) * 1'000'000'000;
    int64_t       measured_ns     = elapsedNs(start, Clock::now());
    while (measured.failed() == 0 && measured.attempted() == final_operations && measured_ns < min_measured_ns) {
        const double remaining_ratio =
            static_cast<double>(min_measured_ns - measured_ns) / static_cast<double>(std::max<int64_t>(1, measured_ns));
        const size_t estimated_extension =
            static_cast<size_t>(std::ceil(static_cast<double>(final_operations) * remaining_ratio * 1.10));
        const size_t extension_operations =
            std::max<size_t>(estimated_extension, std::max<size_t>(64, concurrency * direction_count));

        std::cout << "[transfer] extending measured window by " << extension_operations << " global ops after "
                  << static_cast<double>(measured_ns) / 1e9 << " s" << std::endl;
        const auto extension = runTransferBatch(setup.engine,
                                                options_.transfer_directions,
                                                device_blocks,
                                                host_blocks,
                                                disk_blocks,
                                                concurrency,
                                                extension_operations,
                                                coordinate_cursor,
                                                working_set_blocks,
                                                host_is_working_set);
        coordinate_cursor += (extension_operations + direction_count - 1) / direction_count;
        final_operations += extension_operations;

        for (size_t index = 0; index < working_set_blocks; ++index) {
            measured.visited_working_set[index] =
                measured.visited_working_set[index] || extension.visited_working_set[index];
        }
        for (const auto& [direction, extension_stats] : extension.directions) {
            auto& aggregate = measured.directions.at(direction);
            aggregate.attempted += extension_stats.attempted;
            aggregate.succeeded += extension_stats.succeeded;
            aggregate.failed += extension_stats.failed;
            if (aggregate.first_error.empty() && !extension_stats.first_error.empty()) {
                aggregate.first_error        = extension_stats.first_error;
                aggregate.first_failure_type = extension_stats.first_failure_type;
            }
        }
        measured_ns = elapsedNs(start, Clock::now());
    }

    const size_t attempted = measured.attempted();
    const size_t succeeded = measured.succeeded();
    const size_t failed    = measured.failed();
    const size_t visited   = measured.visitedWorkingSetBlocks();
    const bool   wrapped   = ((final_operations + direction_count - 1) / direction_count) > working_set_blocks;

    std::vector<std::string> actual_strategies;
    if (setup.copy_stats->staged_sm.load(std::memory_order_relaxed) > 0)
        actual_strategies.push_back("staged-sm");
    if (setup.copy_stats->cuda_batch.load(std::memory_order_relaxed) > 0)
        actual_strategies.push_back("batch");
    if (setup.copy_stats->generic.load(std::memory_order_relaxed) > 0)
        actual_strategies.push_back("generic");
    const std::string actual_strategy = actual_strategies.empty()     ? "not-applicable" :
                                        actual_strategies.size() == 1 ? actual_strategies.front() :
                                                                        "mixed";
    writer_.addResolvedConfig("actual_copy_strategy", actual_strategy);

    writer_.setWorkload(seed_, final_operations, attempted, succeeded, failed);
    writer_.setTransferWorkload(
        final_operations, attempted, succeeded, failed, working_set_blocks, working_set_blocks, visited, wrapped);
    writer_.addPhaseNs("pilot", pilot_ns);
    writer_.addPhaseNs("measured", measured_ns);
    writer_.addResolvedConfigInt("resolved_operation_count", final_operations);
    writer_.addResolvedConfigInt("min_measured_seconds", min_seconds);

    const double seconds = static_cast<double>(measured_ns) / 1e9;
    if (succeeded > 0 && measured_ns > 0) {
        const double total_bytes = static_cast<double>(succeeded) * payload_bytes;
        const double throughput  = total_bytes / seconds;
        writer_.addMetric("operations_per_second", static_cast<double>(succeeded) / seconds);
        writer_.addMetric("logical_throughput_bytes_per_second", throughput);
        writer_.addMetric("logical_throughput_gbps", throughput / (1024.0 * 1024.0 * 1024.0));
        writer_.addMetric("total_bytes_transferred", total_bytes);
        writer_.addMetric("avg_ns_per_operation", static_cast<double>(measured_ns) / succeeded);
    }
    for (const auto& [direction, stats] : measured.directions) {
        const std::string prefix = "direction." + direction + ".";
        writer_.addMetric(prefix + "attempted", static_cast<double>(stats.attempted));
        writer_.addMetric(prefix + "succeeded", static_cast<double>(stats.succeeded));
        writer_.addMetric(prefix + "failed", static_cast<double>(stats.failed));
        writer_.addMetric(prefix + "bytes", static_cast<double>(stats.succeeded) * payload_bytes);
        if (seconds > 0) {
            writer_.addMetric(prefix + "throughput_bps",
                              static_cast<double>(stats.succeeded) * payload_bytes / seconds);
        }
        if (!stats.first_error.empty()) {
            writer_.addResolvedConfig(prefix + "first_error", stats.first_error);
            writer_.addResolvedConfig(prefix + "first_failure_type", stats.first_failure_type);
        }
    }
    writer_.addStatistic("sample_unit", 1.0);
    writer_.addStatistic("sample_count", static_cast<double>(succeeded));

    for (size_t member_index = 0; member_index < setup.device_pools.size(); ++member_index) {
        for (const auto block : device_blocks[member_index])
            setup.device_pools[member_index]->free(block);
    }
    for (const auto block : host_blocks)
        host_pool->free(block);
    for (const auto block : disk_blocks)
        disk_pool->free(block);

    const bool strategy_ok = options_.copy_strategy == "auto" || actual_strategy == options_.copy_strategy;
    return attempted == final_operations && succeeded + failed == attempted && failed == 0
           && visited == working_set_blocks && strategy_ok;
}

TransferDescriptor TransferBenchmarkRunner::createDescriptor(const std::string&               direction,
                                                             const MemberDeviceBlocks&        device_blocks,
                                                             const std::vector<BlockIdxType>& host_blocks,
                                                             const std::vector<BlockIdxType>& disk_blocks,
                                                             size_t                           worker_slot_index,
                                                             size_t                           working_set_index,
                                                             bool                             host_is_working_set) {
    std::vector<BlockIdxType> member_blocks;
    member_blocks.reserve(device_blocks.size());
    for (const auto& blocks : device_blocks) {
        member_blocks.push_back(blocks[worker_slot_index]);
    }
    const size_t host_index = host_is_working_set ? working_set_index : worker_slot_index;
    if (direction == "d2h")
        return TransferDescriptor::deviceToHost(0, member_blocks, host_blocks[host_index]);
    if (direction == "h2d")
        return TransferDescriptor::hostToDevice(0, host_blocks[host_index], member_blocks);
    if (direction == "d2disk")
        return TransferDescriptor::deviceToDisk(0, member_blocks, disk_blocks[working_set_index]);
    if (direction == "disk2d")
        return TransferDescriptor::diskToDevice(0, disk_blocks[working_set_index], member_blocks);
    if (direction == "h2disk")
        return TransferDescriptor::hostToDisk(0, host_blocks[host_index], disk_blocks[working_set_index]);
    return TransferDescriptor::diskToHost(0, disk_blocks[working_set_index], host_blocks[host_index]);
}

TransferBenchmarkRunner::BatchResult
TransferBenchmarkRunner::runTransferBatch(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                                          const std::vector<std::string>&                    directions,
                                          const MemberDeviceBlocks&                          device_blocks,
                                          const std::vector<BlockIdxType>&                   host_blocks,
                                          const std::vector<BlockIdxType>&                   disk_blocks,
                                          size_t                                             worker_count,
                                          size_t                                             operation_count,
                                          size_t                                             start_coordinate,
                                          size_t                                             working_set_blocks,
                                          bool                                               host_is_working_set) {
    std::vector<BatchResult> worker_results(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            auto& result = worker_results[worker];
            result.visited_working_set.assign(working_set_blocks, false);
            for (const auto& direction : directions)
                result.directions.emplace(direction, DirectionStats{});
            const auto operations = scheduleTransferWorker(operation_count,
                                                           directions.size(),
                                                           start_coordinate,
                                                           working_set_blocks,
                                                           options_.disk_access_pattern == "random",
                                                           seed_,
                                                           worker_count,
                                                           worker);
            for (const auto& operation : operations) {
                const auto& direction = directions[operation.direction_index];
                auto&       stats     = result.directions.at(direction);
                ++stats.attempted;
                result.visited_working_set[operation.working_set_index] = true;
                auto descriptor                                         = createDescriptor(direction,
                                                   device_blocks,
                                                   host_blocks,
                                                   disk_blocks,
                                                   worker,
                                                   operation.working_set_index,
                                                   host_is_working_set);
                auto context                                            = engine->submit(descriptor);
                if (context->success()) {
                    ++stats.succeeded;
                } else {
                    ++stats.failed;
                    if (stats.first_error.empty()) {
                        const auto error         = context->errorInfo();
                        stats.first_error        = error.ToString();
                        stats.first_failure_type = ErrorCodeToString(error.code());
                    }
                }
            }
        });
    }
    for (auto& worker : workers)
        worker.join();

    BatchResult result;
    result.visited_working_set.assign(working_set_blocks, false);
    for (const auto& direction : directions)
        result.directions.emplace(direction, DirectionStats{});
    for (const auto& worker_result : worker_results) {
        for (size_t index = 0; index < working_set_blocks; ++index)
            result.visited_working_set[index] =
                result.visited_working_set[index] || worker_result.visited_working_set[index];
        for (const auto& [direction, worker_stats] : worker_result.directions) {
            auto& stats = result.directions.at(direction);
            stats.attempted += worker_stats.attempted;
            stats.succeeded += worker_stats.succeeded;
            stats.failed += worker_stats.failed;
            if (stats.first_error.empty() && !worker_stats.first_error.empty()) {
                stats.first_error        = worker_stats.first_error;
                stats.first_failure_type = worker_stats.first_failure_type;
            }
        }
    }
    return result;
}

std::string TransferBenchmarkRunner::createDiskWorkDir() {
    std::string       path = options_.disk_path + "/benchmark_XXXXXX";
    std::vector<char> writable(path.begin(), path.end());
    writable.push_back('\0');
    char* result = ::mkdtemp(writable.data());
    if (result == nullptr)
        throw std::runtime_error("Failed to create disk work dir in " + options_.disk_path);
    return result;
}

}  // namespace rtp_llm::benchmark
