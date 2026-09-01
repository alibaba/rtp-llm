#pragma once

#include <torch/torch.h>

#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm::pp_serialization {

torch::Tensor   serializePlan(const PPExecutionPlan& plan, bool empty_plan);
PPExecutionPlan deserializePlan(const torch::Tensor& buffer);

torch::Tensor  serializeSampleResult(const PPSampleResult& result);
PPSampleResult deserializeSampleResult(const torch::Tensor& buffer);

torch::Tensor         serializeTensorsMetadata(const PPIntermediateTensors& tensors);
PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata);

}  // namespace rtp_llm::pp_serialization
