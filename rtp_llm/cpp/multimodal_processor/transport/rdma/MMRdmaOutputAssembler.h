#pragma once

#include <numeric>
#include <vector>

#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Reassembles tensors read from one or more RDMA slots.

inline bool assembleMMRdmaOutput(const std::vector<torch::Tensor>&        mm_tensors,
                                 const std::vector<MMRdmaTensorPB::Role>& roles,
                                 const MultimodalOutputPB*                output_pb,
                                 MultimodalOutput*                        mm_output) {
    try {
        if (mm_output == nullptr || output_pb == nullptr || mm_tensors.size() != roles.size()) {
            return false;
        }
        std::vector<torch::Tensor> embedding_chunks;
        torch::Tensor              mm_position_id;
        bool                       has_pos_id = false;
        std::vector<torch::Tensor> extra_inputs;
        for (size_t i = 0; i < roles.size(); ++i) {
            switch (roles[i]) {
                case MMRdmaTensorPB::EMBEDDING:
                    embedding_chunks.emplace_back(mm_tensors[i]);
                    break;
                case MMRdmaTensorPB::POS_ID:
                    if (has_pos_id) return false;
                    mm_position_id = mm_tensors[i].to(torch::kCPU);
                    has_pos_id = true;
                    break;
                case MMRdmaTensorPB::EXTRA_INPUT:
                    extra_inputs.emplace_back(mm_tensors[i]);
                    break;
                case MMRdmaTensorPB::ROLE_UNSPECIFIED:
                    RTP_LLM_LOG_WARNING("rdma manifest tensor %zu has unset role; treating as protocol "
                                        "error and falling back to inline bytes",
                                        i);
                    return false;
                default:
                    return false;
            }
        }
        if (embedding_chunks.empty()) return false;
        auto embedding = embedding_chunks.size() == 1 ? embedding_chunks[0] : torch::cat(embedding_chunks, 0);
        std::vector<int64_t> split_sizes(output_pb->split_size().begin(), output_pb->split_size().end());
        const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
        if (split_sizes.empty() || split_total != embedding.size(0)) return false;
        if (has_pos_id && split_total != mm_position_id.size(0)) return false;
        if (!extra_inputs.empty() && extra_inputs.size() != split_sizes.size()) return false;
        MultimodalOutput assembled;
        assembled.mm_features = embedding.split(split_sizes, 0);
        if (has_pos_id) assembled.mm_position_ids = mm_position_id.split(split_sizes, 0);
        if (!extra_inputs.empty()) assembled.mm_extra_input = std::move(extra_inputs);
        *mm_output = std::move(assembled);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("rdma output materialization failed: %s", e.what());
        return false;
    }
}

}  // namespace rtp_llm
