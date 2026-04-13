#pragma once

#include <torch/all.h>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

class NormalOutputDispatcher {
public:
    NormalOutputDispatcher() = default;

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

private:
    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& token_ids_cpu,
                              const torch::Tensor& success_cpu) const;
};

}  // namespace rtp_llm
