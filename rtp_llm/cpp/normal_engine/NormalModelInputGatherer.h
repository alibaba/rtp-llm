#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"

namespace rtp_llm {

struct TensorHolder;

struct NormalModelInputGathererConfig {
    size_t                                     num_layers{};
    size_t                                     vocab_size{};
    size_t                                     input_vocab_size{};
    bool                                       has_positional_encoding{};
    bool                                       is_multimodal{};
    PositionIdsStyle                           mm_position_ids_style{};
    size_t                                     position_id_len_factor{};
    RoleType                                   role_type{};
    bool                                       decode_entrance{};
    size_t                                     block_stride_bytes{};
    size_t                                     scale_stride_bytes{};
    size_t                                     seq_size_per_block{};
    size_t                                     kernel_seq_size_per_block{};
    size_t                                     kernel_blocks_per_kv_block = 1;
    bool                                       use_opaque_kv_cache_store  = false;
    std::unordered_map<std::string, GroupBase> kv_cache_groups;
    bool                                       warm_up{};
    bool                                       enable_detail_log{};
    bool                                       enable_model_inputs_log{};
};

class NormalModelInputGatherer {
public:
    explicit NormalModelInputGatherer(const NormalModelInputGathererConfig& config);

    absl::StatusOr<GptModelInputs> gather(const StreamGroups& stream_groups, TensorHolder& host_holder) const;

    // Build only the CUDA kv_cache_kernel_block_id tensor in 3-D layout.
    // Read-only over streams: no step(), no sibling kv_cache_block_id, no
    // other gather sub-step. Empty input returns an undefined tensor.
    absl::StatusOr<torch::Tensor> gatherKvCacheKernelBlockId(const StreamGroups& stream_groups,
                                                             TensorHolder&       host_holder) const;

private:
    void gatherKvCacheKernelBlockIdToHost(const StreamGroups& stream_groups, torch::Tensor& host_tensor) const;
    GptModelInputs allocateModelInputBuffers(const StreamGroups& stream_groups) const;
    void           initializeKvCacheMetadata(GptModelInputs& model_input) const;
    absl::Status   processDecodeStreams(GptModelInputs& model_input, const StreamGroups& stream_groups) const;
    absl::Status   processContextStreams(GptModelInputs&     model_input,
                                         const StreamGroups& stream_groups,
                                         TensorHolder&       host_holder) const;

    NormalModelInputGathererConfig config_;
    // Canonical sorted cache tags. Entry i of every group-dimension model-input
    // payload belongs to boundary_group_tags_[i]; the consuming model derives the
    // same order from its own CacheConfig.
    std::vector<std::string> boundary_group_tags_;
};

}  // namespace rtp_llm
