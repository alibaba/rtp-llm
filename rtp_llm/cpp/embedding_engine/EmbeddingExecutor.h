#pragma once

#include <memory>
#include <bitset>
#include <torch/python.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/all.h>
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/embedding_engine/ModelRequest.h"
#include "rtp_llm/cpp/engine_base/Executor.h"

namespace rtp_llm {

namespace HandlerArgs {

enum class Arg : uint32_t {
    INPUT_LENGTHS,
    HIDDEN_STATES,
    INPUT_IDS,
    ATTENTION_MASK,
    MOE_GATING,
    // reserve as number marker
    NUM_INPUT_TYPES
};

constexpr size_t NUM_INPUT_TYPES = static_cast<size_t>(Arg::NUM_INPUT_TYPES);

using Flag = std::bitset<NUM_INPUT_TYPES>;

}  // namespace HandlerArgs

class EmbeddingExecutor {
public:
    explicit EmbeddingExecutor(const EngineInitParams& params, rtp_llm::DeviceBase* device, py::object handler);

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

    rtp_llm::DeviceBase*            device_;

private:
    std::unique_ptr<GptModel>       model_;
    py::object                      handler_;
    HandlerArgs::Flag               handler_args_;
    py::handle                      torch_type_;
    rtp_llm::BufferPtr              max_position_ids_buf_;
    kmonitor::MetricsReporterPtr    metrics_reporter_ = nullptr;
    ModelConfig                     model_config_;
    ParallelismConfig               parallelism_config;
    EPLBConfig                      eplb_config;

    ModelRequest                     generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<GptModelInputs>   gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    std::unique_ptr<GptModelOutputs> copyResultToCPU(th::Tensor gpu_outputs) const;
    absl::Status                     updateStreams(py::object                           post_process_output,
                                                   const std::list<EmbeddingStreamPtr>& streams,
                                                   int                                  total_batch_size) const;
    absl::Status
    sliceTensor(py::object gpu_outputs, const std::list<EmbeddingStreamPtr>& streams, int total_batch_size) const;
    absl::Status
    slicePyList(py::object gpu_outputs, const std::list<EmbeddingStreamPtr>& streams, int total_batch_size) const;
    absl::StatusOr<py::object> postProcess(const ModelRequest& model_request, const GptModelOutputs& gpu_outputs);
    void calcTokenNum(const std::list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const;
    void init_position_ids(int max_seq_len);
    void reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const;
};

}  // namespace rtp_llm
