#pragma once

#include <memory>
#include <optional>

#include <torch/all.h>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

std::optional<ErrorInfo> collectStreamSamplerError(const SamplerOutput& sampler_output,
                                                   const torch::Tensor& success_cpu,
                                                   int                  batch_idx_in,
                                                   int                  cur_batch_size);

class NormalOutputDispatcher {
public:
    explicit NormalOutputDispatcher(std::vector<int64_t>                       output_vocab_ids = {},
                                    std::shared_ptr<autil::LockFreeThreadPool> thread_pool      = nullptr):
        output_vocab_ids_(std::move(output_vocab_ids)), thread_pool_(std::move(thread_pool)) {}

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

private:
    bool restoreCurrentTokenIds(const GenerateStreamPtr& stream,
                                torch::Tensor&           batch_token_ids,
                                torch::Tensor&           current_token_ids,
                                size_t                   token_position) const;

    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& success_cpu,
                              const torch::Tensor& batch_custom_output = {},
                              bool                 has_custom_output    = false,
                              bool                 custom_output_failed = false) const;

private:
    std::vector<int64_t>                       output_vocab_ids_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
};

}  // namespace rtp_llm
