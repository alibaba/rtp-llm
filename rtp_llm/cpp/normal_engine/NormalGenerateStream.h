#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rtp_llm {

class NormalGenerateStream: public GenerateStream {
public:
    NormalGenerateStream(const GenerateStream& stream): GenerateStream(stream) {
        CopyOnWrite(stream);
        generate_outputs_queue_.setCapacity(1000);
    }

    NormalGenerateStream(const std::shared_ptr<GenerateInput>& query,
                         const ModelConfig&                    model_config,
                         const RuntimeConfig&                  runtime_config,
                         const ResourceContext&                resource_context,
                         kmonitor::MetricsReporterPtr          metrics_reporter,
                         size_t                                extra_reserve_token_num = 0,
                         bool                                  perf_test               = false):
        GenerateStream(query, model_config, runtime_config, resource_context, metrics_reporter, extra_reserve_token_num, perf_test),
        request_id_(query->request_id) {
        generate_outputs_queue_.setCapacity(1000);
    }

    ~NormalGenerateStream() {
        generate_outputs_queue_.wakeup();
        // Release any nextOutput() waiter parked on our own condition variable.
        output_wait_->wake.store(true, std::memory_order_release);
        output_wait_->cv.notify_all();
    }

    bool                         hasOutput() override;
    ErrorResult<GenerateOutputs> nextOutput() override;
    ErrorResult<GenerateOutputs> nextOutput(const std::function<bool()>& is_cancelled) override;
    void                         updateOutput(const StreamUpdateInfo& update_info) override;

private:
    GenerateOutputs prepareGenerateOutput(const StreamUpdateInfo& update_info);
    void            enqueueGenerateOutput(GenerateOutputs&& generate_results);

    int64_t                                   request_id_{0};
    bool                                      finished_{false};
    autil::SynchronizedQueue<GenerateOutputs> generate_outputs_queue_;

    // Waiter coordination for nextOutput(). Shared ownership follows the same idiom GenerateStream
    // uses for its own cv_ / AsyncBookkeepingCoordinator, so adding these members does not change the
    // stream's implicit copy/move properties. The flag is atomic so a producer never has to take this
    // mutex: enqueueGenerateOutput() runs while the stream mutex_ is held, and taking a second mutex
    // there would add lock nesting to a hot path for no benefit.
    struct OutputWaitCoordinator {
        std::mutex              mu;
        std::condition_variable cv;
        std::atomic<bool>       wake{false};
    };
    std::shared_ptr<OutputWaitCoordinator> output_wait_ = std::make_shared<OutputWaitCoordinator>();
};
}  // namespace rtp_llm
