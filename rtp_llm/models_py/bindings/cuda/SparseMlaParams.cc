#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

using namespace torch_ext;

namespace rtp_llm {

static const int MIN_CACHE_BATCH_SIZE  = 1024;
static const int MIN_CACHE_INPUT_TOKEN = 1024;

void SparseMlaParams::ensureTensorSize(int batch_size, int token_num, bool is_cuda_graph, bool is_capture) {
    int old_max_batch_size = max_batch_size_;
    int old_max_token_num  = max_token_num_;

    max_batch_size_ = std::max(max_batch_size_, std::max(batch_size, MIN_CACHE_BATCH_SIZE));
    max_token_num_  = std::max(max_token_num_, std::max(token_num, MIN_CACHE_INPUT_TOKEN));

    std::vector<std::vector<int64_t>> shapes = {
        {max_token_num_},  // expanded_seq_lens
        {max_token_num_},  // topk_indices_offset
        {max_token_num_},  // ks
        {max_token_num_}   // ke
    };

    size_t total_i32_elements = 0;
    for (const auto& shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        size = (size + 31) / 32 * 32;  // alignTo32
        total_i32_elements += size;
    }

    bool need_realloc = (total_i32_elements > max_i32_elements_);

    // Check if reallocation is needed in CUDA graph replay mode (not capture mode)
    // During capture phase, reallocation is allowed to set up the graph
    if (need_realloc && is_cuda_graph && !is_capture) {
        RTP_LLM_LOG_ERROR("[SparseMlaParams] Buffer reallocation required in CUDA graph mode, which is not allowed!");
        RTP_LLM_LOG_ERROR("Reallocation reason:");
        RTP_LLM_LOG_ERROR("  Current max_i32_elements: %zu", max_i32_elements_);
        RTP_LLM_LOG_ERROR("  Required i32_elements: %zu", total_i32_elements);
        RTP_LLM_LOG_ERROR("Parameter changes:");
        if (old_max_batch_size != max_batch_size_) {
            RTP_LLM_LOG_ERROR(
                "  - max_batch_size: %d -> %d (requested: %d)", old_max_batch_size, max_batch_size_, batch_size);
        }
        if (old_max_token_num != max_token_num_) {
            RTP_LLM_LOG_ERROR(
                "  - max_token_num: %d -> %d (requested: %d)", old_max_token_num, max_token_num_, token_num);
        }
        RTP_LLM_LOG_ERROR("Tensor sizes breakdown:");
        RTP_LLM_LOG_ERROR("  - expanded_seq_lens: %d elements", max_token_num_);
        RTP_LLM_LOG_ERROR("  - topk_indices_offset: %d elements", max_token_num_);
        RTP_LLM_LOG_ERROR("  - ks: %d elements", max_token_num_);
        RTP_LLM_LOG_ERROR("  - ke: %d elements", max_token_num_);
        throw std::runtime_error("[SparseMlaParams] Buffer reallocation required in CUDA graph replay mode");
    }

    if (!need_realloc && buf_h_i32_.defined() && buf_d_i32_.defined()) {
        return;
    }

    max_i32_elements_ = total_i32_elements;

    // Use base class allocateManyBuffer
    auto alloc_ret_h = FlashInferMlaAttnParams::allocateManyBuffer(shapes, false, torch::kInt32);
    buf_h_i32_       = std::get<0>(alloc_ret_h);
    auto& tensors_h  = std::get<1>(alloc_ret_h);

    auto alloc_ret_d = FlashInferMlaAttnParams::allocateManyBuffer(shapes, true, torch::kInt32);
    buf_d_i32_       = std::get<0>(alloc_ret_d);
    auto& tensors_d  = std::get<1>(alloc_ret_d);

    expanded_seq_lens_h_   = tensors_h[0];
    topk_indices_offset_h_ = tensors_h[1];
    ks_h_                  = tensors_h[2];
    ke_h_                  = tensors_h[3];

    expanded_seq_lens_d_   = tensors_d[0];
    topk_indices_offset_d_ = tensors_d[1];
    ks_d_                  = tensors_d[2];
    ke_d_                  = tensors_d[3];
}

void SparseMlaParams::fillParamsInternal(bool                 is_prefill,
                                         const torch::Tensor& input_lengths_cpu,
                                         const torch::Tensor& prefix_lengths_cpu,
                                         const torch::Tensor& sequence_lengths_cpu,
                                         int                  batch_size,
                                         int                  seq_size_per_block,
                                         int64_t              total_tokens,
                                         const torch::Tensor& positions_h) {
    if (is_prefill) {
        const auto input_lengths_ptr  = input_lengths_cpu.data_ptr<int32_t>();
        const auto prefix_lengths_ptr = prefix_lengths_cpu.defined() && prefix_lengths_cpu.numel() > 0 ?
                                            prefix_lengths_cpu.data_ptr<int32_t>() :
                                            nullptr;
        const auto positions_ptr      = positions_h.data_ptr<int32_t>();

        auto expanded_seq_lens_ptr   = expanded_seq_lens_h_.data_ptr<int32_t>();
        auto topk_indices_offset_ptr = topk_indices_offset_h_.data_ptr<int32_t>();
        auto ks_ptr                  = ks_h_.data_ptr<int32_t>();
        auto ke_ptr                  = ke_h_.data_ptr<int32_t>();

        int64_t offset   = 0;
        int64_t k_offset = 0;
        for (int i = 0; i < batch_size; ++i) {
            const int32_t input_len  = input_lengths_ptr[i];
            const int32_t prefix_len = prefix_lengths_ptr ? prefix_lengths_ptr[i] : 0;
            const int32_t kv_len     = input_len + prefix_len;

            for (int j = 0; j < input_len; ++j) {
                const int32_t seq_len_value     = kv_len - input_len + 1 + j;
                expanded_seq_lens_ptr[offset]   = seq_len_value;
                topk_indices_offset_ptr[offset] = 0;
                ks_ptr[offset]                  = k_offset;
                ke_ptr[offset]                  = k_offset + seq_len_value;
                offset += 1;
            }
            k_offset += kv_len;
        }
    }
}

void SparseMlaParams::refreshBuffer(int batch_size, int token_num, bool is_prefill) {
    if (!buf_h_i32_.defined() || !buf_d_i32_.defined()) {
        return;
    }
    cudaStream_t stream      = GET_CURRENT_STREAM();
    size_t       total_bytes = buf_h_i32_.numel() * sizeof(int32_t);
    cudaMemcpyAsync(buf_d_i32_.data_ptr(), buf_h_i32_.data_ptr(), total_bytes, cudaMemcpyHostToDevice, stream);

    std::vector<int64_t> shape;
    if (is_prefill) {
        shape = {token_num};
        expanded_seq_lens_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        expanded_seq_lens_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        topk_indices_offset_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        topk_indices_offset_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        ks_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        ks_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        ke_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        ke_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    } else {
        shape = {batch_size};
        expanded_seq_lens_h_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        expanded_seq_lens_d_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        topk_indices_offset_h_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        topk_indices_offset_d_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        ks_h_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        ks_d_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        ke_h_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        ke_d_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
    }
}

void SparseMlaParams::fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
    // Step 1: Call base class fillParams to fill shared parameters
    // Call base class method to fill shared parameters
    FlashInferMlaAttnParams::fillParams(attn_inputs.prefix_lengths,
                                        attn_inputs.sequence_lengths,
                                        attn_inputs.input_lengths,
                                        attn_inputs.kv_cache_block_id_host,
                                        seq_size_per_block,
                                        attn_inputs.is_cuda_graph,
                                        attn_inputs.is_capture);

    // Step 2: Fill IndexerParams-specific parameters
    bool is_prefill = attn_inputs.is_prefill;
    int  batch_size = is_prefill ? attn_inputs.input_lengths.size(0) : attn_inputs.sequence_lengths.size(0);

    // Now we can directly access base class positions_h, batch_indice_h, kvlen_d, etc.

    int64_t total_tokens = 0;
    if (is_prefill) {
        const auto input_lengths_ptr = attn_inputs.input_lengths.data_ptr<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            total_tokens += input_lengths_ptr[i];
        }

        if (total_tokens > 0) {
            ensureTensorSize(
                batch_size, static_cast<int>(total_tokens), attn_inputs.is_cuda_graph, attn_inputs.is_capture);

            // Use base class positions_h (no need to pass from parameter)
            fillParamsInternal(true,
                               attn_inputs.input_lengths,
                               attn_inputs.prefix_lengths,
                               attn_inputs.sequence_lengths,
                               batch_size,
                               seq_size_per_block,
                               total_tokens,
                               positions_h);
            refreshBuffer(batch_size, static_cast<int>(total_tokens), true);

            expanded_seq_lens   = expanded_seq_lens_d_;
            topk_indices_offset = topk_indices_offset_d_;
            ks                  = ks_d_;
            ke                  = ke_d_;
        } else {
            auto options_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            expanded_seq_lens   = torch::empty({0}, options_cuda);
            topk_indices_offset = torch::empty({0}, options_cuda);
            ks                  = torch::empty({0}, options_cuda);
            ke                  = torch::empty({0}, options_cuda);
        }
    } else {
        expanded_seq_lens = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        auto options_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        topk_indices_offset = torch::empty({0}, options_cuda);
        ks                  = torch::empty({0}, options_cuda);
        ke                  = torch::empty({0}, options_cuda);

        if (batch_size != 0) {
            ensureTensorSize(batch_size, batch_size, attn_inputs.is_cuda_graph, attn_inputs.is_capture);

            // Use base class positions_h (no need to pass from parameter)
            fillParamsInternal(false,
                               attn_inputs.input_lengths,
                               attn_inputs.prefix_lengths,
                               attn_inputs.sequence_lengths,
                               batch_size,
                               seq_size_per_block,
                               0,
                               positions_h);
            refreshBuffer(batch_size, batch_size, false);
        }
        // In decode mode, expanded_seq_lens equals kvlen_d (sequence_lengths + 1)
        expanded_seq_lens = kvlen_d;
    }
}

void registerPySparseMlaParams(pybind11::module& m) {
    // Third template parameter must be FlashInferMlaAttnParams (not ParamsBase)
    // This allows Python to access all def_readonly attributes of the base class
    pybind11::class_<SparseMlaParams, std::shared_ptr<SparseMlaParams>, FlashInferMlaAttnParams>(m, "SparseMlaParams")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::SparseMlaParams& self, torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
                self.fillParams(attn_inputs, seq_size_per_block);
            },
            pybind11::arg("attention_inputs"),
            pybind11::arg("seq_size_per_block"))
        .def_readonly("expanded_seq_lens", &SparseMlaParams::expanded_seq_lens)
        .def_readonly("topk_indices_offset", &SparseMlaParams::topk_indices_offset)
        .def_readonly("ks", &SparseMlaParams::ks)
        .def_readonly("ke", &SparseMlaParams::ke)
        .def_readwrite("schedule_metadata", &SparseMlaParams::schedule_metadata);

    m.def(
        "prepare_sparse_mla_params",
        [](torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
            auto params = std::make_shared<rtp_llm::SparseMlaParams>();
            params->fillParams(attn_inputs, seq_size_per_block);
            return params;
        },
        pybind11::arg("attention_inputs"),
        pybind11::arg("seq_size_per_block"));
}

}  // namespace rtp_llm
