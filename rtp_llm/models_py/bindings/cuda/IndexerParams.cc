#include "rtp_llm/models_py/bindings/cuda/IndexerParams.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

using namespace torch_ext;

namespace rtp_llm {

static const size_t MIN_CACHE_I32_ELEMENTS = 1 << 20;  // 1M int32 elements
static const size_t MIN_CACHE_I64_ELEMENTS = 1 << 20;  // 1M int64 elements
static const int    MIN_CACHE_BATCH_SIZE   = 256;
static const int    MIN_CACHE_INPUT_TOKEN  = 512;
static const int    MIN_CACHE_MAX_SEQ_LEN  = 512;

// TODO: cuda graph need max buffer to avoid reallocation
std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexerParams::allocateManyBuffer(
    const std::vector<std::vector<int64_t>>& shapes, bool is_device, torch::ScalarType dtype) {
    std::vector<torch::Tensor> tensors;
    std::vector<size_t>        sizes;
    size_t                     total_size = 0;

    for (const auto& shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        size = (size + 31) / 32 * 32;
        sizes.push_back(size);
        total_size += size;
    }

    torch::Tensor        buf;
    torch::TensorOptions options;
    if (is_device) {
        options = torch::dtype(dtype).device(torch::kCUDA).requires_grad(false);
        buf     = torch::empty({static_cast<int64_t>(total_size)}, options);
    } else {
        options = torch::dtype(dtype).device(torch::kCPU).requires_grad(false).pinned_memory(true);
        buf     = torch::empty({static_cast<int64_t>(total_size)}, options);
    }

    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        size_t actual_size = 1;
        for (const auto dim : shapes[i]) {
            actual_size *= dim;
        }
        tensors.emplace_back(buf.slice(0, offset, offset + actual_size).reshape(shapes[i]));
        offset += sizes[i];
    }

    return {buf, tensors};
}

static torch::Tensor toCudaInt32(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return torch::Tensor();
    }
    if (tensor.device().is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    return tensor.to(torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
}

static size_t alignTo32(size_t value) {
    return (value + 31) / 32 * 32;
}

void IndexerParams::ensureTensorSize(int batch_size, int token_num, int max_seq_len) {
    max_batch_size_ = std::max(max_batch_size_, std::max(batch_size, MIN_CACHE_BATCH_SIZE));
    max_token_num_  = std::max(max_token_num_, std::max(token_num, MIN_CACHE_INPUT_TOKEN));
    max_seq_len_    = std::max(max_seq_len_, std::max(max_seq_len, MIN_CACHE_MAX_SEQ_LEN));

    std::vector<std::vector<int64_t>> shapes = {
        {max_token_num_},                // batch_indice
        {max_token_num_},                // positions
        {max_token_num_},                // expanded_seq_lens
        {max_token_num_},                // topk_indices_offset
        {max_token_num_},                // ks
        {max_token_num_},                // ke
        {max_batch_size_},               // decode batch_indice
        {max_batch_size_},               // decode positions
        {max_batch_size_, max_seq_len_}  // page_table_1
    };

    size_t total_i32_elements = 0;
    for (const auto& shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        total_i32_elements += alignTo32(size);
    }

    size_t required_i32_elements = std::max(alignTo32(total_i32_elements), MIN_CACHE_I32_ELEMENTS);
    size_t required_i64_elements =
        std::max(alignTo32(static_cast<size_t>(std::max(max_token_num_, max_batch_size_))), MIN_CACHE_I64_ELEMENTS);

    bool need_realloc = (required_i32_elements > max_i32_elements_) || (required_i64_elements > max_i64_elements_);
    if (!need_realloc && buf_h_i32_.defined() && buf_d_i32_.defined() && buf_h_i64_.defined()) {
        return;
    }

    max_i32_elements_ = required_i32_elements;
    max_i64_elements_ = required_i64_elements;

    auto alloc_ret_h = allocateManyBuffer(shapes, false, torch::kInt32);
    buf_h_i32_       = std::get<0>(alloc_ret_h);
    auto& tensors_h  = std::get<1>(alloc_ret_h);

    auto alloc_ret_d = allocateManyBuffer(shapes, true, torch::kInt32);
    buf_d_i32_       = std::get<0>(alloc_ret_d);
    auto& tensors_d  = std::get<1>(alloc_ret_d);

    batch_indice_h_        = tensors_h[0];
    positions_h_           = tensors_h[1];
    expanded_seq_lens_h_   = tensors_h[2];
    topk_indices_offset_h_ = tensors_h[3];
    ks_h_                  = tensors_h[4];
    ke_h_                  = tensors_h[5];
    page_table_1_h_        = tensors_h[8];

    batch_indice_d_        = tensors_d[0];
    positions_d_           = tensors_d[1];
    expanded_seq_lens_d_   = tensors_d[2];
    topk_indices_offset_d_ = tensors_d[3];
    ks_d_                  = tensors_d[4];
    ke_d_                  = tensors_d[5];
    page_table_1_d_        = tensors_d[8];

    auto alloc_ret_i64 = allocateManyBuffer({{static_cast<int64_t>(max_i64_elements_)}}, false, torch::kInt64);
    buf_h_i64_         = std::get<0>(alloc_ret_i64);
}

void IndexerParams::fillParamsInternal(bool                 is_prefill,
                                       const torch::Tensor& input_lengths_cpu,
                                       const torch::Tensor& prefix_lengths_cpu,
                                       const torch::Tensor& sequence_lengths_cpu,
                                       int                  seq_size_per_block,
                                       int64_t              total_tokens,
                                       int64_t              max_seq_len,
                                       torch::Tensor&       slot_mapping_h) {
    if (is_prefill) {
        const auto input_lengths_ptr  = input_lengths_cpu.data_ptr<int32_t>();
        const auto prefix_lengths_ptr = prefix_lengths_cpu.defined() && prefix_lengths_cpu.numel() > 0 ?
                                            prefix_lengths_cpu.data_ptr<int32_t>() :
                                            nullptr;

        auto batch_indice_ptr        = batch_indice_h_.data_ptr<int32_t>();
        auto positions_ptr           = positions_h_.data_ptr<int32_t>();
        auto expanded_seq_lens_ptr   = expanded_seq_lens_h_.data_ptr<int32_t>();
        auto topk_indices_offset_ptr = topk_indices_offset_h_.data_ptr<int32_t>();
        auto ks_ptr                  = ks_h_.data_ptr<int32_t>();
        auto ke_ptr                  = ke_h_.data_ptr<int32_t>();

        int64_t offset   = 0;
        int64_t k_offset = 0;
        int64_t q_offset = 0;
        for (int i = 0; i < batch_size; ++i) {
            const int32_t input_len  = input_lengths_ptr[i];
            const int32_t prefix_len = prefix_lengths_ptr ? prefix_lengths_ptr[i] : 0;
            const int32_t kv_len     = input_len + prefix_len;

            for (int j = 0; j < input_len; ++j) {
                const int32_t seq_len_value     = kv_len - input_len + 1 + j;
                batch_indice_ptr[offset]        = i;
                positions_ptr[offset]           = j + prefix_len;
                expanded_seq_lens_ptr[offset]   = seq_len_value;
                topk_indices_offset_ptr[offset] = q_offset;
                ks_ptr[offset]                  = k_offset;
                ke_ptr[offset]                  = k_offset + seq_len_value;
                offset += 1;
            }
            q_offset += input_len;
            k_offset += kv_len;
        }

        slot_mapping_h = buf_h_i64_.slice(0, 0, total_tokens).reshape({total_tokens});
    } else {
        auto batch_indice_ptr = batch_indice_h_.data_ptr<int32_t>();
        auto positions_ptr    = positions_h_.data_ptr<int32_t>();
        auto seq_lens_ptr     = sequence_lengths_cpu.data_ptr<int32_t>();

        for (int i = 0; i < batch_size; ++i) {
            batch_indice_ptr[i] = i;
            positions_ptr[i]    = seq_lens_ptr[i];
        }
        auto page_table_ptr = page_table_1_h_.data_ptr<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            int64_t row_offset = static_cast<int64_t>(i) * max_seq_len;
            for (int j = 0; j < max_seq_len; ++j) {
                page_table_ptr[row_offset + j] = j;
            }
        }

        slot_mapping_h = buf_h_i64_.slice(0, 0, batch_size).reshape({batch_size});
    }
}

void IndexerParams::refreshBuffer(int batch_size, int token_num, int max_seq_len, bool is_prefill) {
    if (!buf_h_i32_.defined() || !buf_d_i32_.defined()) {
        return;
    }
    cudaStream_t stream      = GET_CURRENT_STREAM();
    size_t       total_bytes = buf_h_i32_.numel() * sizeof(int32_t);
    cudaMemcpyAsync(buf_d_i32_.data_ptr(), buf_h_i32_.data_ptr(), total_bytes, cudaMemcpyHostToDevice, stream);

    std::vector<int64_t> shape;
    if (is_prefill) {
        shape = {token_num};
        batch_indice_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        batch_indice_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        positions_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        positions_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
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
        batch_indice_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        batch_indice_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        positions_h_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
        positions_d_.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
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

void IndexerParams::fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
    is_prefill  = attn_inputs.is_prefill;
    block_table = attn_inputs.kv_cache_block_id_device;

    torch::Tensor input_lengths_cpu    = attn_inputs.input_lengths;
    torch::Tensor prefix_lengths_cpu   = attn_inputs.prefix_lengths;
    torch::Tensor sequence_lengths_cpu = attn_inputs.sequence_lengths;

    batch_size = is_prefill ? input_lengths_cpu.size(0) : sequence_lengths_cpu.size(0);

    torch::Tensor batch_indice_d;
    torch::Tensor batch_indice_h;
    torch::Tensor positions_h;
    torch::Tensor slot_mapping_h;

    if (is_prefill) {
        seq_lens      = toCudaInt32(attn_inputs.input_lengths);
        cu_q_seqlens  = attn_inputs.cu_seqlens;
        cu_kv_seqlens = attn_inputs.cu_kv_seqlens;

        int64_t    total_tokens      = 0;
        const auto input_lengths_ptr = input_lengths_cpu.data_ptr<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            total_tokens += input_lengths_ptr[i];
        }

        if (total_tokens > 0) {
            ensureTensorSize(batch_size, static_cast<int>(total_tokens), 0);

            fillParamsInternal(true,
                               input_lengths_cpu,
                               prefix_lengths_cpu,
                               sequence_lengths_cpu,
                               seq_size_per_block,
                               total_tokens,
                               0,
                               slot_mapping_h);
            refreshBuffer(batch_size, static_cast<int>(total_tokens), 0, true);

            batch_indice_h      = batch_indice_h_;
            positions_h         = positions_h_;
            batch_indice_d      = batch_indice_d_;
            positions_d         = positions_d_;
            expanded_seq_lens   = expanded_seq_lens_d_;
            topk_indices_offset = topk_indices_offset_d_;
            ks                  = ks_d_;
            ke                  = ke_d_;
        } else {
            auto options_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            batch_indice_d      = torch::empty({0}, options_cuda);
            expanded_seq_lens   = torch::empty({0}, options_cuda);
            topk_indices_offset = torch::empty({0}, options_cuda);
            ks                  = torch::empty({0}, options_cuda);
            ke                  = torch::empty({0}, options_cuda);
            positions_d         = torch::empty({0}, options_cuda);
            batch_indice_h      = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            positions_h         = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            slot_mapping_h      = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        }

        page_table_1 = page_table_1_d_;
    } else {
        seq_lens          = toCudaInt32(attn_inputs.sequence_lengths) + 1;
        cu_q_seqlens      = attn_inputs.decode_cu_seqlens_d;
        cu_kv_seqlens     = attn_inputs.cu_kv_seqlens;
        expanded_seq_lens = seq_lens;

        auto options_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        topk_indices_offset = torch::empty({0}, options_cuda);
        ks                  = torch::empty({0}, options_cuda);
        ke                  = torch::empty({0}, options_cuda);

        if (batch_size == 0) {
            batch_indice_d = torch::empty({0}, options_cuda);
            positions_d    = torch::empty({0}, options_cuda);
            page_table_1   = torch::empty({0, 0}, options_cuda);
            batch_indice_h = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            positions_h    = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            slot_mapping_h = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        } else {
            const int64_t max_seq_len =
                sequence_lengths_cpu.numel() > 0 ? (sequence_lengths_cpu.max().item<int64_t>() + 1) : 0;
            ensureTensorSize(batch_size, batch_size, static_cast<int>(max_seq_len));

            fillParamsInternal(false,
                               input_lengths_cpu,
                               prefix_lengths_cpu,
                               sequence_lengths_cpu,
                               seq_size_per_block,
                               0,
                               max_seq_len,
                               slot_mapping_h);
            refreshBuffer(batch_size, batch_size, static_cast<int>(max_seq_len), false);

            batch_indice_h = batch_indice_h_;
            positions_h    = positions_h_;
            batch_indice_d = batch_indice_d_;
            positions_d    = positions_d_;
            page_table_1   = page_table_1_d_;
        }
    }

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

        slot_mapping =
            torch::empty({positions_h.numel()}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        cudaStream_t stream = GET_CURRENT_STREAM();
        cudaMemcpyAsync(slot_mapping.data_ptr(),
                        slot_mapping_h.data_ptr(),
                        slot_mapping_h.numel() * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        stream);
    } else {
        slot_mapping = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    }
}

void registerPyIndexerParams(pybind11::module& m) {
    pybind11::class_<IndexerParams, std::shared_ptr<IndexerParams>, rtp_llm::ParamsBase>(m, "IndexerParams")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::IndexerParams& self, torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
                self.fillParams(attn_inputs, seq_size_per_block);
            },
            pybind11::arg("attention_inputs"),
            pybind11::arg("seq_size_per_block"))
        .def_readonly("expanded_seq_lens", &IndexerParams::expanded_seq_lens)
        .def_readonly("page_table_1", &IndexerParams::page_table_1)
        .def_readonly("cu_q_seqlens", &IndexerParams::cu_q_seqlens)
        .def_readonly("cu_kv_seqlens", &IndexerParams::cu_kv_seqlens)
        .def_readonly("topk_indices_offset", &IndexerParams::topk_indices_offset)
        .def_readonly("batch_size", &IndexerParams::batch_size)
        .def_readonly("seq_lens", &IndexerParams::seq_lens)
        .def_readonly("positions_d", &IndexerParams::positions_d)
        .def_readonly("slot_mapping", &IndexerParams::slot_mapping)
        .def_readonly("block_table", &IndexerParams::block_table)
        .def_readonly("ks", &IndexerParams::ks)
        .def_readonly("ke", &IndexerParams::ke)
        .def_readwrite("schedule_metadata", &IndexerParams::schedule_metadata)
        .def_readonly("is_prefill", &IndexerParams::is_prefill);

    m.def(
        "prepare_indexer_params",
        [](torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block) {
            auto params = std::make_shared<rtp_llm::IndexerParams>();
            params->fillParams(attn_inputs, seq_size_per_block);
            return params;
        },
        pybind11::arg("attention_inputs"),
        pybind11::arg("seq_size_per_block"));
}

}  // namespace rtp_llm
