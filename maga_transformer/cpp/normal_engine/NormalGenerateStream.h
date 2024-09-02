#pragma once
#include "maga_transformer/cpp/dataclass/GenerateStream.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class NormalGenerateStream: public GenerateStream {
public:
    NormalGenerateStream(const GenerateStream&  stream): GenerateStream(stream) {
        CopyOnWrite(stream);
    }

    NormalGenerateStream(const std::shared_ptr<GenerateInput>& query,
                         const ft::GptInitParameter&           params,
                         const ResourceContext&                resource_context,
                         kmonitor::MetricsReporterPtr          metrics_reporter):
        GenerateStream(query, params, resource_context, metrics_reporter) {
        generate_outputs_             = std::make_shared<GenerateOutputs>();
        generate_outputs_->request_id = query->request_id;
        generate_outputs_queue_.setCapacity(1000);
    }

    ~NormalGenerateStream() {
        generate_outputs_queue_.wakeup();
    }

public:
    absl::StatusOr<GenerateOutputs> nextOutput() override;

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs) override;

protected:
    std::shared_ptr<GenerateOutputs>          generate_outputs_;
    autil::SynchronizedQueue<GenerateOutputs> generate_outputs_queue_;
};
}  // namespace rtp_llm
