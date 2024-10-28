#pragma once
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include <cstdint>

namespace ft = fastertransformer;

namespace rtp_llm {

class NormalGenerateStream: public GenerateStream {
public:
    NormalGenerateStream(const GenerateStream& stream): GenerateStream(stream) {
        CopyOnWrite(stream);
        generate_outputs_queue_.setCapacity(1000);
    }

    NormalGenerateStream(const std::shared_ptr<GenerateInput>& query,
                         const ft::GptInitParameter&           params,
                         const ResourceContext&                resource_context,
                         kmonitor::MetricsReporterPtr          metrics_reporter):
        GenerateStream(query, params, resource_context, metrics_reporter), request_id_(query->request_id) {
        generate_outputs_queue_.setCapacity(1000);
    }

    ~NormalGenerateStream() {
        generate_outputs_queue_.wakeup();
    }

    ErrorResult<GenerateOutputs> nextOutput() override;

    bool hasOutput() override;

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs,
                      const ft::BufferPtr& all_probs,
                      const ft::BufferPtr& loss,
                      bool                 update_queue = true) override;

private:
    GenerateOutputs prepareGenerateOutput(const ft::BufferPtr& new_tokens,
                                          const ft::BufferPtr& hidden_states,
                                          const ft::BufferPtr& logits,
                                          const ft::BufferPtr& cum_log_probs,
                                          const ft::BufferPtr& all_probs,
                                          const ft::BufferPtr& loss);
    void            enqueueGenerateOutput(GenerateOutputs generate_results);

    int64_t                                   request_id_{0};
    bool                                      finished_{false};
    autil::SynchronizedQueue<GenerateOutputs> generate_outputs_queue_;
};
}  // namespace rtp_llm
