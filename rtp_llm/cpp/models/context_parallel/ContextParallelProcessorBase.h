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
    explicit IContextParallelProcessor(const ParallelismConfig& parallelism_config):
        parallelism_config_(parallelism_config) {}
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
};

class ContextParallelProcessorFactory {
public:
    static std::unique_ptr<IContextParallelProcessor> create(ProcessorType            type,
                                                             const ParallelismConfig& parallelism_config);
};

}  // namespace rtp_llm
