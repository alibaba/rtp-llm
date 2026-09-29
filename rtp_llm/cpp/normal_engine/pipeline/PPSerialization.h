#pragma once

#include <torch/torch.h>

#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm::pp_serialization {

/** Plan/result tensors decode to independent pinned CPU storage; callers handle device placement. */
torch::Tensor   serializePlan(const PPExecutionPlan& plan, bool metadata_only);
PPExecutionPlan deserializePlan(const torch::Tensor& buffer);

torch::Tensor     serializeExecutionResult(const PPExecutionResult& result);
PPExecutionResult deserializeExecutionResult(const torch::Tensor& buffer);

torch::Tensor serializeTensorsMetadata(const PPIntermediateTensors& tensors);

/** Decode metadata into receive views backed by reusable storage for a fixed stage boundary.
 * Previous consumers must finish before the storage is reused. */
PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata, PPIntermediateTensors& backing_buffers);

}  // namespace rtp_llm::pp_serialization
