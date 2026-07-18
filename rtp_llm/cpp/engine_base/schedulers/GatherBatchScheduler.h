#pragma once
#include <unordered_set>
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"

namespace rtp_llm {

struct GatherBatchSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
    }
    uint32_t batch_size_;
};

// GatherBatchScheduler is used to gather batch of streams, and sort streams by streamId.
// Currently it is only used in CI with prompt_batch input, which may occur unstable result
class GatherBatchScheduler: virtual public FIFOScheduler {
public:
    explicit GatherBatchScheduler(const RuntimeConfig&                   runtime_config,
                                  const ModelConfig&                     model_config,
                                  const PDSepConfig&                     pd_sep_config,
                                  const ParallelismConfig&               parallelism_config,
                                  const ModelSpecificConfig&             model_specific_config,
                                  const std::shared_ptr<KVCacheManager>& cache_manager,
                                  const kmonitor::MetricsReporterPtr     metrics_reporter,
                                  const int                              max_score_len = 1):
        FIFOScheduler(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter,
                      max_score_len) {
        RTP_LLM_LOG_INFO("GatherBatchScheduler init");
        gather_batch_size_ = 1;
    }

    void updateSchedulerInfo(const std::string& scheduler_info) override {
        GatherBatchSchedulerConfigLocal config;
        autil::legacy::FromJsonString(config, scheduler_info);
        {
            std::lock_guard<std::mutex> lock(lock_);
            gather_batch_size_ = config.batch_size_;
        }
        cond_.notify_all();
        RTP_LLM_LOG_INFO("GatherBatchScheduler update batch size to %d", gather_batch_size_);
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return gatherCountStreams(waiting_) >= static_cast<size_t>(gather_batch_size_) || !running_.empty()
                   || !loading_.empty();
        });

        // running: advance + cleanup
        for (auto it = running_.begin(); it != running_.end();) {
            it->advance();
            if (!it->alive()) {
                it = running_.erase(it);
            } else {
                ++it;
            }
        }

        // loading: check ready -> activate -> move to running
        for (auto it = loading_.begin(); it != loading_.end();) {
            if (it->isReady()) {
                it->activate();
                if (it->alive()) {
                    running_.splice(running_.end(), loading_, it++);
                } else {
                    it = loading_.erase(it);
                }
            } else {
                ++it;
            }
        }

        const bool python_model_busy = !running_.empty();
        if (gatherCountStreams(waiting_) >= static_cast<size_t>(gather_batch_size_) && !python_model_busy) {
            std::list<GenerateStreamPtr> new_streams;
            for (auto& unit : waiting_) {
                for (auto& s : unit.streams) {
                    if (!s->hasError()) {
                        new_streams.push_back(s);
                    }
                    if (new_streams.size() >= static_cast<size_t>(gather_batch_size_)) {
                        break;
                    }
                }
                if (new_streams.size() >= static_cast<size_t>(gather_batch_size_)) {
                    break;
                }
            }
            if (new_streams.size() >= static_cast<size_t>(gather_batch_size_)) {
                std::unordered_set<int64_t> scheduled_ids;
                for (auto& stream : new_streams) {
                    stream->prepare();
                    while (stream->alive() && !stream->isReady()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    if (stream->alive()) {
                        stream->activate();
                    }
                }
                new_streams.remove_if([](const auto& s) { return !s->alive(); });
                new_streams.sort([](const GenerateStreamPtr& a, const GenerateStreamPtr& b) {
                    return a->streamId() < b->streamId();
                });
                for (auto& s : new_streams) {
                    scheduled_ids.insert(s->streamId());
                }
                // Move scheduled streams into a running unit
                ScheduleUnit run_unit;
                run_unit.group_id = -1;
                for (auto& s : new_streams) {
                    run_unit.streams.push_back(s);
                }
                running_.push_back(std::move(run_unit));
                // Remove scheduled streams from waiting_
                for (auto it = waiting_.begin(); it != waiting_.end();) {
                    for (auto sit = it->streams.begin(); sit != it->streams.end();) {
                        if (scheduled_ids.count((*sit)->streamId())) {
                            sit = it->streams.erase(sit);
                        } else {
                            ++sit;
                        }
                    }
                    if (it->streams.empty()) {
                        it = waiting_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            gather_batch_size_ = 1;
        }

        return gatherFlattenRunning();
    }

protected:
    size_t gatherCountStreams(const std::list<ScheduleUnit>& queue) const {
        size_t total = 0;
        for (const auto& unit : queue) {
            total += unit.size();
        }
        return total;
    }

    std::list<GenerateStreamPtr> gatherFlattenRunning() const {
        std::list<GenerateStreamPtr> result;
        for (const auto& unit : running_) {
            for (const auto& stream : unit.streams) {
                result.push_back(stream);
            }
        }
        return result;
    }

    int gather_batch_size_;
};

}  // namespace rtp_llm