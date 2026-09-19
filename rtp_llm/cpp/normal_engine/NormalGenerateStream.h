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
        // Release any nextOutput() waiter parked on our own condition variable, publishing the flag
        // under the wait mutex for the same reason enqueueGenerateOutput() does: a bare store plus
        // notify can miss a waiter that is between its predicate evaluation and its park.
        //
        // This is a courtesy wake, NOT an ownership guarantee. It is not safe to destroy a stream
        // while a consumer is inside nextOutput(); callers hold a GenerateStreamPtr for the whole
        // call, so the stream and this coordinator outlive any waiter.
        {
            std::lock_guard<std::mutex> wait_lock(output_wait_->mu);
            output_wait_->wake.store(true, std::memory_order_relaxed);
        }
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
    // stream's implicit copy/move properties.
    //
    // LOCK ORDER: stream mutex_ -> mu -> the output queue's own lock. Both the producer
    // (enqueueGenerateOutput, which runs under mutex_) and the destructor publish `wake` while holding
    // mu, and the consumer clears it, re-tests the queue and registers the wait in one mu critical
    // section. That serialization is what makes a lost notification impossible; publishing the flag
    // without mu would let a notify land after a waiter's predicate returned false but before the wait
    // was registered. The consumer never takes mutex_ under mu (its cancellation callback and
    // checkTimeout() both run outside mu), and the producer never holds the queue lock while taking mu,
    // so the order is acyclic. `wake` stays atomic for readability; every access is under mu.
    struct OutputWaitCoordinator {
        std::mutex              mu;
        std::condition_variable cv;
        std::atomic<bool>       wake{false};
    };
    std::shared_ptr<OutputWaitCoordinator> output_wait_ = std::make_shared<OutputWaitCoordinator>();
};
}  // namespace rtp_llm
