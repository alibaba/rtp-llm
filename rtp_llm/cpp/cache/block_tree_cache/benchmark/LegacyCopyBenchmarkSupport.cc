#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/LegacyCopyBenchmarkSupport.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/LegacyTransferEngine.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <chrono>
namespace rtp_llm::benchmark {
using Clock = std::chrono::steady_clock;
int64_t elapsedNs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
class RecordingDeviceHostCopyStrategy final: public DeviceHostCopyStrategy {
public:
    RecordingDeviceHostCopyStrategy(std::unique_ptr<DeviceHostCopyStrategy> delegate,
                                    std::atomic<size_t>*                    completed,
                                    BenchmarkDeviceHostCopyStats*           stats):
        delegate_(std::move(delegate)), completed_(completed), stats_(stats) {}

    StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                              const DeviceHostCopyOptions&          options,
                              const DeviceHostCopyExecutionContext& context) override {
        const auto start   = Clock::now();
        auto       result  = delegate_->tryExecute(plan, options, context);
        const auto elapsed = elapsedNs(start, Clock::now());
        if (result.status == StrategyStatus::DONE) {
            completed_->fetch_add(1, std::memory_order_relaxed);
            stats_->lowest_api_ns.fetch_add(elapsed, std::memory_order_relaxed);
            stats_->lowest_api_calls.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

private:
    std::unique_ptr<DeviceHostCopyStrategy> delegate_;
    std::atomic<size_t>*                    completed_;
    BenchmarkDeviceHostCopyStats*           stats_;
};

class RecordingCuda3DBatchDeviceHostCopyStrategy final: public Cuda3DBatchDeviceHostCopyStrategy {
public:
    RecordingCuda3DBatchDeviceHostCopyStrategy(Cuda3DBatchDeviceHostCopyStrategy&& strategy,
                                               BenchmarkDeviceHostCopyStats*       stats):
        Cuda3DBatchDeviceHostCopyStrategy(std::move(strategy)), stats_(stats) {}

    StrategyResult tryExecute(const std::vector<HostBufferView>&     hosts,
                              const std::vector<TransferDescriptor>& descriptors,
                              const std::vector<const GroupSet*>&    group_sets,
                              const DeviceHostCopyExecutionContext&  context) override {
        const auto start   = Clock::now();
        auto       result  = Cuda3DBatchDeviceHostCopyStrategy::tryExecute(hosts, descriptors, group_sets, context);
        const auto elapsed = elapsedNs(start, Clock::now());
        if (result.status == StrategyStatus::DONE) {
            stats_->cuda_3d_batch.fetch_add(1, std::memory_order_relaxed);
            stats_->lowest_api_ns.fetch_add(elapsed, std::memory_order_relaxed);
            stats_->lowest_api_calls.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

private:
    BenchmarkDeviceHostCopyStats* stats_;
};

void installStrategyRecorders(PerRankBlockTransferEngine&                          engine,
                              const std::shared_ptr<BenchmarkDeviceHostCopyStats>& stats) {
    auto& strategies = engine.legacy_->device_host_executor_->strategies_;
    for (auto& strategy : strategies) {
        if (auto* copy3d = dynamic_cast<Cuda3DBatchDeviceHostCopyStrategy*>(strategy.get())) {
            strategy = std::make_unique<RecordingCuda3DBatchDeviceHostCopyStrategy>(std::move(*copy3d), stats.get());
            continue;
        }
        std::atomic<size_t>* completed = nullptr;
        if (dynamic_cast<StagedSmDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->staged_sm;
        } else if (dynamic_cast<CudaBatchDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->cuda_batch;
        } else if (dynamic_cast<GenericMultiCopyDeviceHostCopyStrategy*>(strategy.get()) != nullptr) {
            completed = &stats->generic;
        }
        RTP_LLM_CHECK(completed != nullptr);
        strategy = std::make_unique<RecordingDeviceHostCopyStrategy>(std::move(strategy), completed, stats.get());
    }
}

void recordLegacyStaging(PerRankBlockTransferEngine& engine, BenchmarkJsonWriter& writer, bool is_swa) {
    if (engine.legacy_->device_disk_executor_ != nullptr) {
        const size_t full_capacity =
            engine.legacy_->device_disk_executor_->batchCapacity(rtp_llm::CacheGroupType::FULL);
        const size_t swa_capacity = engine.legacy_->device_disk_executor_->batchCapacity(rtp_llm::CacheGroupType::SWA);
        writer.addResolvedConfigInt("disk_to_device_full_batch_capacity", full_capacity);
        writer.addResolvedConfigInt("disk_to_device_swa_batch_capacity", swa_capacity);
        writer.addResolvedConfigInt("disk_to_device_batch_capacity", is_swa ? swa_capacity : full_capacity);
    }
}
}  // namespace rtp_llm::benchmark
