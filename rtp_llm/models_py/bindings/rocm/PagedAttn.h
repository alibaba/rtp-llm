#pragma once
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/utils/RopeConfig.h"
#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/rocm/FMHARocmBase.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/devices/DeviceData.h"

namespace py = pybind11;
using namespace torch_ext;
namespace rtp_llm {

struct forward_param {
    torch::Tensor out;          // Output tensor for attention results
    torch::Tensor exp_sums;     // Exponential sums for attention computation
    torch::Tensor max_logits;   // Maximum logits for numerical stability
    torch::Tensor tmp_out;      // Temporary output buffer
    torch::Tensor query;        // Query tensor after preprocessing
    int64_t num_kv_heads;      // Number of key/value attention heads
    double scale;              // Attention scale factor
    torch::Tensor context_lens; // Context lengths for each sequence
    int64_t block_size;        // Size of each KV cache block
    int64_t max_seq_len;       // Maximum sequence length
    int64_t partition_size;    // Size of partitions for paged attention
};

// struct PagedAttnParams {
//     // Basic parameters for attention dimensions
//     size_t batch_size = 0;            // Number of sequences in the batch
//     size_t local_head_num = 0;        // Number of attention heads
//     size_t size_per_head = 0;         // Size of each attention head
//     size_t kv_head_num = 0;           // Number of key/value heads (may differ from query heads)
//     size_t max_blocks_per_batch = 0;  // Maximum number of KV cache blocks per batch
//     size_t decoder_batch_size = 0;    // Batch size for decoder
//     int64_t max_seq_len = 0;          // Maximum sequence length
//     int64_t max_prefix_length = 0;    // Maximum prefix prompt length

//     // Stream and tensor buffers
//     int64_t stream_ = 0;              // CUDA stream for computation
//     torch::Tensor sequence_lengths;    // Lengths of each sequence
//     torch::Tensor prefix_prompt_lengths; // Lengths of prefix prompts
//     torch::Tensor output;             // Output tensor
//     torch::Tensor cu_seqlens;         // Cumulative sequence lengths
//     torch::Tensor cu_kv_seqlens;      // Cumulative KV sequence lengths

//     // Attention specific parameters
//     bool decode_plan = false;         // Whether this is a decode operation
//     rtp_llm::DataType attn_type;      // Data type for attention computation
//     rtp_llm::KVBlockArray kv_block_array;      // KV cache block array structure
//     // Weights and configuration
//     struct {
//         torch::Tensor* qkv_weight;
//     } weights;
// };

// using PagedAttnParamsPtr = std::shared_ptr<PagedAttnParams>;

class PagedAttnDecodeOp : public rtp_llm::FMHARocmBase {
public:
    PagedAttnDecodeOp(
        const GptInitParameter& gpt_init_parameter
        // const DeviceInitParams& device_init_params
    );
    bool          support(torch_ext::PyAttentionInputs attn_inputs);

    CKAttnPtr prepare(torch_ext::PyAttentionInputs attn_inputs);
    forward_param forward(
        const torch::Tensor& qkv,
        FMHAType fmha_type,
        std::optional<torch_ext::KVCache> kv_cache,
        const CKAttnPtr& params
    );

private:
    // Offset for KV cache blocks, calculated as num_layers * block_nums
    size_t kv_block_offset_;
    // Flag to control whether to use AITER paged attention, controlled by USE_AITER_PA env var
    bool use_aiter_pa_ = true;
};

// Register the PagedAttnDecodeOp class with Python bindings
void registerPagedAttnDecodeOp(py::module& m);

}