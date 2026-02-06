#include "rtp_llm/models_py/bindings/cuda/SparseMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

using namespace torch_ext;

namespace rtp_llm {

static const size_t MIN_CACHE_I32_ELEMENTS = 1 << 20;  // 1M int32 elements
static const size_t MIN_CACHE_I64_ELEMENTS = 1 << 20;  // 1M int64 elements
static const int    MIN_CACHE_BATCH_SIZE   = 1024;
static const int    MIN_CACHE_INPUT_TOKEN  = 1024;
static const int    MIN_CACHE_MAX_SEQ_LEN  = 8192;

void SparseMlaParams::ensureTensorSize(int batch_size, int token_num, int max_seq_len) {
    max_batch_size_ = std::max(max_batch_size_, std::max(batch_size, MIN_CACHE_BATCH_SIZE));
    max_token_num_  = std::max(max_token_num_, std::max(token_num, MIN_CACHE_INPUT_TOKEN));
    max_seq_len_    = std::max(max_seq_len_, std::max(max_seq_len, MIN_CACHE_MAX_SEQ_LEN));

    std::vector<std::vector<int64_t>> shapes = {
        {max_token_num_},                // expanded_seq_lens
        {max_token_num_},                // topk_indices_offset
        {max_token_num_},                // ks
        {max_token_num_},                // ke
        {max_batch_size_, max_seq_len_}  // page_table_1
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

    size_t required_i32_elements = std::max(total_i32_elements, MIN_CACHE_I32_ELEMENTS);
    size_t required_i64_elements = std::max(
        (static_cast<size_t>(std::max(max_token_num_, max_batch_size_)) + 31) / 32 * 32, MIN_CACHE_I64_ELEMENTS);

    bool need_realloc = (required_i32_elements > max_i32_elements_) || (required_i64_elements > max_i64_elements_);
    if (!need_realloc && buf_h_i32_.defined() && buf_d_i32_.defined() && buf_h_i64_.defined() && buf_d_i64_.defined()) {
        return;
    }

    max_i32_elements_ = required_i32_elements;
    max_i64_elements_ = required_i64_elements;

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
    page_table_1_h_        = tensors_h[4];

    expanded_seq_lens_d_   = tensors_d[0];
    topk_indices_offset_d_ = tensors_d[1];
    ks_d_                  = tensors_d[2];
    ke_d_                  = tensors_d[3];
    page_table_1_d_        = tensors_d[4];

    auto alloc_ret_i64_h =
        FlashInferMlaAttnParams::allocateManyBuffer({{static_cast<int64_t>(max_i64_elements_)}}, false, torch::kInt64);
    buf_h_i64_      = std::get<0>(alloc_ret_i64_h);
    slot_mapping_h_ = std::get<1>(alloc_ret_i64_h)[0];

    auto alloc_ret_i64_d =
        FlashInferMlaAttnParams::allocateManyBuffer({{static_cast<int64_t>(max_i64_elements_)}}, true, torch::kInt64);
    buf_d_i64_      = std::get<0>(alloc_ret_i64_d);
    slot_mapping_d_ = std::get<1>(alloc_ret_i64_d)[0];
}

void SparseMlaParams::fillParamsInternal(bool                 is_prefill,
                                         const torch::Tensor& input_lengths_cpu,
                                         const torch::Tensor& prefix_lengths_cpu,
                                         const torch::Tensor& sequence_lengths_cpu,
                                         int                  batch_size,
                                         int                  seq_size_per_block,
                                         int64_t              total_tokens,
                                         int64_t              max_seq_len,
                                         const torch::Tensor& positions_h,
                                         torch::Tensor&       slot_mapping_h) {
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

        slot_mapping_h_ = buf_h_i64_.slice(0, 0, total_tokens).reshape({total_tokens});
        slot_mapping_h  = slot_mapping_h_;
    } else {
        auto page_table_ptr = page_table_1_h_.data_ptr<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            int64_t row_offset = static_cast<int64_t>(i) * max_seq_len;
            for (int j = 0; j < max_seq_len; ++j) {
                page_table_ptr[row_offset + j] = j;
            }
        }

        slot_mapping_h_ = buf_h_i64_.slice(0, 0, batch_size).reshape({batch_size});
        slot_mapping_h  = slot_mapping_h_;
    }
}

void SparseMlaParams::refreshBuffer(int batch_size, int token_num, int max_seq_len, bool is_prefill) {
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
        page_table_1_h_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
        page_table_1_d_.unsafeGetTensorImpl()->set_sizes_contiguous({0});
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
        page_table_1_h_.unsafeGetTensorImpl()->set_sizes_contiguous({batch_size, max_seq_len});
        page_table_1_d_.unsafeGetTensorImpl()->set_sizes_contiguous({batch_size, max_seq_len});
    }
}

void SparseMlaParams::fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
    // Step 1: Call base class fillParams to fill shared parameters
    torch::Tensor prefix_lengths_cpu   = attn_inputs.prefix_lengths;
    torch::Tensor sequence_lengths_cpu = attn_inputs.sequence_lengths;
    torch::Tensor input_lengths_cpu    = attn_inputs.input_lengths;
    torch::Tensor block_table          = attn_inputs.kv_cache_block_id_device;

    // Convert block_table to CPU (base class needs HOST version)
    torch::Tensor kv_cache_block_id_host =
        block_table.defined() && block_table.numel() > 0 ?
            block_table.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32)) :
            torch::Tensor();

    // Call base class method to fill shared parameters
    FlashInferMlaAttnParams::fillParams(
        prefix_lengths_cpu, sequence_lengths_cpu, input_lengths_cpu, kv_cache_block_id_host, seq_size_per_block);

    // Step 2: Fill IndexerParams-specific parameters
    bool is_prefill = attn_inputs.is_prefill;
    int  batch_size = is_prefill ? input_lengths_cpu.size(0) : sequence_lengths_cpu.size(0);

    // Now we can directly access base class positions_h, batch_indice_h, kvlen_d, etc.
    torch::Tensor slot_mapping_h;

    int64_t total_tokens = 0;
    if (is_prefill) {
        const auto input_lengths_ptr = input_lengths_cpu.data_ptr<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            total_tokens += input_lengths_ptr[i];
        }

        if (total_tokens > 0) {
            ensureTensorSize(batch_size, static_cast<int>(total_tokens), 0);

            // Use base class positions_h (no need to pass from parameter)
            fillParamsInternal(true,
                               input_lengths_cpu,
                               prefix_lengths_cpu,
                               sequence_lengths_cpu,
                               batch_size,
                               seq_size_per_block,
                               total_tokens,
                               0,
                               positions_h,
                               slot_mapping_h);
            refreshBuffer(batch_size, static_cast<int>(total_tokens), 0, true);

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
            slot_mapping_h      = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        }

        page_table_1 = page_table_1_d_;
    } else {
        expanded_seq_lens = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        auto options_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        topk_indices_offset = torch::empty({0}, options_cuda);
        ks                  = torch::empty({0}, options_cuda);
        ke                  = torch::empty({0}, options_cuda);

        if (batch_size == 0) {
            page_table_1   = torch::empty({0, 0}, options_cuda);
            slot_mapping_h = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        } else {
            const int64_t max_seq_len =
                sequence_lengths_cpu.numel() > 0 ? (sequence_lengths_cpu.max().item<int64_t>() + 1) : 0;
            ensureTensorSize(batch_size, batch_size, static_cast<int>(max_seq_len));

            // Use base class positions_h (no need to pass from parameter)
            fillParamsInternal(false,
                               input_lengths_cpu,
                               prefix_lengths_cpu,
                               sequence_lengths_cpu,
                               batch_size,
                               seq_size_per_block,
                               0,
                               max_seq_len,
                               positions_h,
                               slot_mapping_h);
            refreshBuffer(batch_size, batch_size, static_cast<int>(max_seq_len), false);

            page_table_1 = page_table_1_d_;
        }
        // In decode mode, expanded_seq_lens equals kvlen_d (sequence_lengths + 1)
        expanded_seq_lens = kvlen_d;
    }

    // Step 3: Calculate slot_mapping (using base class batch_indice_h)
    if (block_table.defined() && block_table.numel() > 0 && positions_h.defined() && positions_h.numel() > 0) {
        auto          block_table_cpu = block_table.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
        const int64_t max_blocks      = block_table_cpu.size(1);
        auto          block_table_ptr = block_table_cpu.data_ptr<int32_t>();
        auto          batch_indice_ptr = batch_indice_h.data_ptr<int32_t>();
        auto          positions_ptr    = positions_h.data_ptr<int32_t>();
        auto          slot_mapping_ptr = slot_mapping_h.data_ptr<int64_t>();

        for (int64_t i = 0; i < positions_h.numel(); ++i) {
            const int32_t batch_id     = batch_indice_ptr[i];
            const int32_t position     = positions_ptr[i];
            const int32_t block_index  = position / seq_size_per_block;
            const int32_t block_offset = position % seq_size_per_block;
            const int32_t block_number = block_table_ptr[batch_id * max_blocks + block_index];
            slot_mapping_ptr[i]        = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
        }

        const int64_t slot_mapping_numel = slot_mapping_h.numel();
        cudaStream_t  stream             = GET_CURRENT_STREAM();
        if (slot_mapping_numel > 0) {
            size_t total_bytes = static_cast<size_t>(slot_mapping_numel) * sizeof(int64_t);
            cudaMemcpyAsync(
                slot_mapping_d_.data_ptr(), slot_mapping_h.data_ptr(), total_bytes, cudaMemcpyHostToDevice, stream);
        }
        slot_mapping_h_.unsafeGetTensorImpl()->set_sizes_contiguous({slot_mapping_numel});
        slot_mapping_d_.unsafeGetTensorImpl()->set_sizes_contiguous({slot_mapping_numel});
        slot_mapping = slot_mapping_d_;
    } else {
        slot_mapping = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
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
        .def_readonly("page_table_1", &SparseMlaParams::page_table_1)
        .def_readonly("topk_indices_offset", &SparseMlaParams::topk_indices_offset)
        .def_readonly("slot_mapping", &SparseMlaParams::slot_mapping)
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
