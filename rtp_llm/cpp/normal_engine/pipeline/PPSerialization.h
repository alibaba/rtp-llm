#pragma once

#include <torch/torch.h>

#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm::pp_serialization {

torch::Tensor   serializePlan(const PPExecutionPlan& plan, bool empty_plan);
PPExecutionPlan deserializePlan(const torch::Tensor& buffer);

torch::Tensor     serializeExecutionResult(const PPExecutionResult& result);
PPExecutionResult deserializeExecutionResult(const torch::Tensor& buffer);

/** Per-chunk status carried in the activation-metadata header.
 *  `ok` leaves the V4 path's behaviour identical. `rejected` tells the consumer that the
 *  producer's admission failed for this chunk: no payload follows, the chunk must not
 *  advance any consumer frontier, and the request terminates with an error. This is what
 *  lets a producer-side rejection reach a peer that is about to post receives, once the
 *  per-chunk WORLD agreement no longer makes both stages reject together. */
enum class PPTensorsStatus : uint8_t {
    ok       = 0,
    rejected = 1,
};

torch::Tensor         serializeTensorsMetadata(const PPIntermediateTensors& tensors,
                                               PPTensorsStatus              status = PPTensorsStatus::ok);
PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata, PPTensorsStatus* status = nullptr);

}  // namespace rtp_llm::pp_serialization
