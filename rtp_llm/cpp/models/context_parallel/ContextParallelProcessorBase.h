#pragma once
#include <memory>
#include <vector>
#include <torch/extension.h>
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace torch_ext {
struct PyContextParallelParams;
}

namespace rtp_llm {

struct GptModelInputs;

enum class ProcessorType {
    ZIG_ZAG,
    // Future extensions: ROUND_ROBIN, BLOCK_WISE, etc.
};

class IContextParallelProcessor {
public:
    /// @param cp_chunk_alignment Additional alignment factor on top of (cp_size * 2).
    ///        Pure attention models can use 1. Hybrid models with linear attention
    ///        (FLA chunked algorithm) must pass the FLA chunk_size (64) so that
    ///        each rank's half-segment is a chunk_size multiple — this is required
    ///        for segment-internal h states to coincide with full-sequence chunk
    ///        boundaries, enabling correct SSM cache writes.
    explicit IContextParallelProcessor(const ParallelismConfig& parallelism_config, int cp_chunk_alignment = 1):
        parallelism_config_(parallelism_config), cp_chunk_alignment_(cp_chunk_alignment) {}
    virtual ~IContextParallelProcessor() = default;

    /// @brief Prepare context parallel inputs: split and shuffle tokens, compute restore indices and masks.
    void handleInputs(GptModelInputs& model_input, torch_ext::PyContextParallelParams& cp_params);

    /// @brief Gather outputs from all CP ranks and restore original token order.
    virtual size_t handleOutputs(torch::Tensor&                            hidden_states,
                                 const GptModelInputs&                     inputs,
                                 const torch_ext::PyContextParallelParams& cp_params) = 0;

protected:
    /// @brief Process and distribute input tokens across context parallel ranks
    ///
    /// Plans the distribution of input tokens by splitting and padding as needed.
    /// Each rank receives a processed chunk according to the processing strategy.
    virtual bool plan(const std::vector<int>& total_input_tokens,
                      std::vector<int>&       input_tokens,
                      std::vector<int>&       shuffle_indices,
                      int                     cp_rank,
                      int                     cp_size,
                      int                     cp_chunk_size,
                      int                     cp_padding_size) = 0;

    /// @brief Generate indices to restore original QKV order after parallel processing
    virtual torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) = 0;

    /// @brief Generate padding mask for QKV tensors
    virtual torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                 const torch::Tensor& prefill_cp_padding_lengths,
                                                 int                  cp_size) = 0;

    ParallelismConfig parallelism_config_;
    int               cp_chunk_alignment_ = 1;
};

class ContextParallelProcessorFactory {
public:
    /// @param cp_chunk_alignment See IContextParallelProcessor ctor doc. Hybrid
    ///        models with linear attention should pass 64 (FLA chunk_size).
    static std::unique_ptr<IContextParallelProcessor>
    create(ProcessorType type, const ParallelismConfig& parallelism_config, int cp_chunk_alignment = 1);
};

}  // namespace rtp_llm
