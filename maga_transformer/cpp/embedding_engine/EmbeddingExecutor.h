#pragma once

#include <memory>
#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"

namespace rtp_llm {

class EmbeddingExecutor{
public:
    explicit EmbeddingExecutor(const EngineInitParams& params, ft::DeviceBase* device, py::object handler);

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

private:
    std::unique_ptr<GptModel>             model_;
    std::unique_ptr<ParallelModelWrapper> model_wrapper_;
    py::object                            handler_;
    ft::DeviceBase*                       device_;
    ft::BufferPtr                         max_position_ids_buf_;
    kmonitor::MetricsReporterPtr          metrics_reporter_ = nullptr;
    const fastertransformer::GptInitParameter params_;

    bool                                  use_new_device_impl_ = false;

    ModelRequest                     generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<GptModelInputs>   gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    std::unique_ptr<GptModelOutputs> copyResultToCPU(th::Tensor gpu_outputs) const;
    absl::Status                     updateStreams(th::Tensor    gpu_outputs,
                                                   const std::list<EmbeddingStreamPtr>& streams) const;
    absl::Status                     createAttentionMask(GptModelInputs& model_input) const;
    absl::StatusOr<th::Tensor>       postProcess(const ModelRequest& model_request, const GptModelOutputs& gpu_outputs);
    void calcTokenNum(const std::list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const;    
    void                             init_position_ids(int max_seq_len);
    void reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const;
};
}  // namespace rtp_llm
