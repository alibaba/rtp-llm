#pragma once

#include <torch/torch.h>

#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm::pp_serialization {

/** Plan/result tensors decode to independent pinned CPU storage; callers handle device placement. */
torch::Tensor   serializePlan(const PPExecutionPlan& plan, bool metadata_only);
PPExecutionPlan deserializePlan(const torch::Tensor& buffer);

torch::Tensor     serializeExecutionResult(const PPExecutionResult& result);
PPExecutionResult deserializeExecutionResult(const torch::Tensor& buffer);

/** Metadata decoding allocates CPU/CUDA receive buffers; tensor data is transferred separately. */
torch::Tensor         serializeTensorsMetadata(const PPIntermediateTensors& tensors);
PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata);

}  // namespace rtp_llm::pp_serialization
