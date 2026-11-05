#pragma once
#include <c10/util/FunctionRef.h>
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
    explicit IContextParallelProcessor(const ParallelismConfig& parallelism_config,
                                       bool                     prefer_local_hidden_states = false):
        parallelism_config_(parallelism_config), prefer_local_hidden_states_(prefer_local_hidden_states) {}
    virtual ~IContextParallelProcessor() = default;

    /// @brief Prepare context parallel inputs: split and shuffle tokens, compute restore indices and masks.
    void handleInputs(GptModelInputs& model_input, torch_ext::PyContextParallelParams& cp_params);

    // Capture only: run local work and agree on success before the next collective.
    using LocalOutputStage = c10::function_ref<bool(c10::function_ref<void()>)>;

    /// @brief Gather outputs from all CP ranks and restore original token order.
    virtual size_t handleOutputs(torch::Tensor&                            hidden_states,
                                 const GptModelInputs&                     inputs,
                                 const torch_ext::PyContextParallelParams& cp_params,
                                 LocalOutputStage                          stage = {}) = 0;

    /// @brief CP exit variant that materializes only the last-token rows lm_head needs.
    ///
    /// Instead of all-gathering the full [seq, hidden] sequence and restoring its
    /// original order (handleOutputs), this gathers only the rows selected by
    /// inputs.lm_output_indexes. Each selected row lives on exactly one CP rank
    /// (zigzag is a bijection on valid positions), so the cross-rank merge is a
    /// single all-reduce-sum over a [num_lm, hidden] buffer. On return,
    /// hidden_states is replaced by that [num_lm, hidden] tensor, ordered to match
    /// lm_output_indexes (identical to index_select(restored_hidden, lm_output_indexes)).
    ///
    /// Only valid when no consumer needs the full sequence hidden. Requests
    /// that require all logits or all hidden states are rejected before they
    /// enter a prefill-CP engine.
    virtual void handleOutputsLastHidden(torch::Tensor&                            hidden_states,
                                         const GptModelInputs&                     inputs,
                                         const torch_ext::PyContextParallelParams& cp_params) = 0;

protected:
    /** Valid shuffle indices must be strictly increasing for per-token and multimodal remapping.
     * Indices outside [0, input_length) represent padding. CPU callers provide a zero-filled
     * destination. CUDA callers pass two null pointers and use the indices to gather on device. */
    virtual bool plan(int               input_length,
                      const int*        cpu_src,
                      int*              cpu_dst,
                      std::vector<int>& shuffle_indices,
                      int               cp_rank,
                      int               cp_size,
                      int               cp_chunk_size,
                      int               cp_padding_size) = 0;

    /// @brief Generate indices to restore original QKV order after parallel processing
    virtual torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) = 0;

    /// @brief Generate padding mask for QKV tensors
    virtual torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                 const torch::Tensor& prefill_cp_padding_lengths,
                                                 int                  cp_size) = 0;

    ParallelismConfig parallelism_config_;
    // Used only when global and local row counts are equal. DeepSeek's MTP
    // buffer is rank-local, while ordinary model output is global.
    const bool prefer_local_hidden_states_;
};

class ContextParallelProcessorFactory {
public:
    static std::unique_ptr<IContextParallelProcessor>
    create(ProcessorType type, const ParallelismConfig& parallelism_config, bool prefer_local_hidden_states = false);
};

}  // namespace rtp_llm
