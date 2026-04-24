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

void SparseMlaParams::ensureTensorSize(int batch_size, int token_num, bool forbid_realloc) {
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

    // prepare_cuda_graph (replay only) forbids realloc; capture/init allow it
    if (need_realloc && forbid_realloc) {
        RTP_LLM_LOG_ERROR(
            "[SparseMlaParams] Buffer reallocation required in CUDA graph replay mode, which is not allowed!");
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

void SparseMlaParams::ensureCpTensorSize(int max_idx_count, int batch_size) {
    max_idx_count = std::max(max_idx_count, 1024);
    batch_size    = std::max(batch_size, 256);

    // int64 tensors: kv_restore_unpad_indices + total_global_ids + total_local_ids
    std::vector<std::vector<int64_t>> i64_shapes = {
        {(int64_t)max_idx_count},
        {(int64_t)max_idx_count},
        {(int64_t)max_idx_count},
    };
    size_t total_i64 = 0;
    for (const auto& s : i64_shapes) {
        size_t n = 1;
        for (auto d : s) n *= d;
        n = (n + 31) / 32 * 32;
        total_i64 += n;
    }

    // int32 tensor: cu_kv_seqlens_global(batch_size + 1)
    std::vector<std::vector<int64_t>> i32_shapes = {
        {(int64_t)(batch_size + 1)},
    };
    size_t total_i32 = 0;
    for (const auto& s : i32_shapes) {
        size_t n = 1;
        for (auto d : s) n *= d;
        n = (n + 31) / 32 * 32;
        total_i32 += n;
    }

    bool need_i64_realloc = (total_i64 > cp_max_i64_elements_);
    bool need_i32_realloc = (total_i32 > cp_max_i32_elements_);

    if (need_i64_realloc || !cp_buf_h_i64_.defined()) {
        cp_max_i64_elements_ = total_i64;
        cp_max_idx_count_    = max_idx_count;

        auto alloc_h = FlashInferMlaAttnParams::allocateManyBuffer(i64_shapes, false, torch::kInt64);
        cp_buf_h_i64_ = std::get<0>(alloc_h);
        auto& th = std::get<1>(alloc_h);

        auto alloc_d = FlashInferMlaAttnParams::allocateManyBuffer(i64_shapes, true, torch::kInt64);
        cp_buf_d_i64_ = std::get<0>(alloc_d);
        auto& td = std::get<1>(alloc_d);

        cp_kv_restore_unpad_indices_h_ = th[0];
        cp_total_global_ids_h_         = th[1];
        cp_total_local_ids_h_          = th[2];

        cp_kv_restore_unpad_indices_d_ = td[0];
        cp_total_global_ids_d_         = td[1];
        cp_total_local_ids_d_          = td[2];
    }

    if (need_i32_realloc || !cp_buf_h_i32_2_.defined()) {
        cp_max_i32_elements_  = total_i32;
        cp_max_batch_size_cp_ = batch_size;

        auto alloc_h = FlashInferMlaAttnParams::allocateManyBuffer(i32_shapes, false, torch::kInt32);
        cp_buf_h_i32_2_ = std::get<0>(alloc_h);
        auto& th2 = std::get<1>(alloc_h);

        auto alloc_d = FlashInferMlaAttnParams::allocateManyBuffer(i32_shapes, true, torch::kInt32);
        cp_buf_d_i32_2_ = std::get<0>(alloc_d);
        auto& td2 = std::get<1>(alloc_d);

        cp_cu_kv_seqlens_global_h_ = th2[0];
        cp_cu_kv_seqlens_global_d_ = td2[0];
    }
}

void SparseMlaParams::refreshCpBuffer(int kv_restore_count, int total_ids_count, int batch_size) {
    cudaStream_t stream = GET_CURRENT_STREAM();

    // Copy only the actually-filled portion of each buffer, not the full
    // pre-allocated capacity (which is at least 1024 * 3 int64 elements).
    // Buffer layout (i64): [kv_restore | global_ids | local_ids] — contiguous
    // with 32-element alignment between sub-tensors.
    if (cp_buf_h_i64_.defined() && cp_buf_d_i64_.defined()
        && (kv_restore_count > 0 || total_ids_count > 0)) {
        auto* buf_start = cp_buf_h_i64_.data_ptr<int64_t>();
        const int64_t* last_data_end;
        if (total_ids_count > 0) {
            last_data_end = cp_total_local_ids_h_.data_ptr<int64_t>() + total_ids_count;
        } else {
            last_data_end = cp_kv_restore_unpad_indices_h_.data_ptr<int64_t>() + kv_restore_count;
        }
        size_t i64_bytes = static_cast<size_t>(last_data_end - buf_start) * sizeof(int64_t);
        cudaMemcpyAsync(cp_buf_d_i64_.data_ptr(), buf_start, i64_bytes, cudaMemcpyHostToDevice, stream);
    }

    if (cp_buf_h_i32_2_.defined() && cp_buf_d_i32_2_.defined()) {
        size_t i32_bytes = static_cast<size_t>(batch_size + 1) * sizeof(int32_t);
        cudaMemcpyAsync(cp_buf_d_i32_2_.data_ptr(), cp_buf_h_i32_2_.data_ptr(),
                        i32_bytes, cudaMemcpyHostToDevice, stream);
    }

    std::vector<int64_t> kv_shape  = {(int64_t)kv_restore_count};
    std::vector<int64_t> ids_shape = {(int64_t)total_ids_count};
    std::vector<int64_t> cu_shape  = {(int64_t)(batch_size + 1)};

    cp_kv_restore_unpad_indices_h_.unsafeGetTensorImpl()->set_sizes_contiguous(kv_shape);
    cp_kv_restore_unpad_indices_d_.unsafeGetTensorImpl()->set_sizes_contiguous(kv_shape);
    cp_total_global_ids_h_.unsafeGetTensorImpl()->set_sizes_contiguous(ids_shape);
    cp_total_global_ids_d_.unsafeGetTensorImpl()->set_sizes_contiguous(ids_shape);
    cp_total_local_ids_h_.unsafeGetTensorImpl()->set_sizes_contiguous(ids_shape);
    cp_total_local_ids_d_.unsafeGetTensorImpl()->set_sizes_contiguous(ids_shape);
    cp_cu_kv_seqlens_global_h_.unsafeGetTensorImpl()->set_sizes_contiguous(cu_shape);
    cp_cu_kv_seqlens_global_d_.unsafeGetTensorImpl()->set_sizes_contiguous(cu_shape);

    cp_kv_restore_unpad_indices = cp_kv_restore_unpad_indices_d_;
    cp_total_global_ids         = cp_total_global_ids_d_;
    cp_total_local_ids          = cp_total_local_ids_d_;
    cp_cu_kv_seqlens_global     = cp_cu_kv_seqlens_global_d_;
}

void SparseMlaParams::fillCpPlanParams(const torch::Tensor&         padding_mask,
                                       const torch::Tensor&         kv_restore_indices,
                                       const std::vector<int64_t>&  q0_idx,
                                       const std::vector<int64_t>&  q1_idx,
                                       int                          cp_rank,
                                       int                          local_tokens,
                                       const torch::Tensor&         actual_input_lengths,
                                       const torch::Tensor&         prefix_lengths) {
    const int padded_total  = padding_mask.size(0);
    const int batch_size    = actual_input_lengths.size(0);
    const int max_idx_count = std::max(padded_total, (int)(q0_idx.size() + q1_idx.size()));

    ensureCpTensorSize(max_idx_count, batch_size);

    const auto* pm_ptr = padding_mask.data_ptr<int32_t>();

    // kv_restore_indices may be int32 (from ZigZagProcessor) or int64 — read into a local int64 vector
    std::vector<int64_t> ri_vec(padded_total);
    if (kv_restore_indices.scalar_type() == torch::kInt32) {
        const auto* ri32 = kv_restore_indices.data_ptr<int32_t>();
        for (int i = 0; i < padded_total; ++i) ri_vec[i] = static_cast<int64_t>(ri32[i]);
    } else {
        const auto* ri64 = kv_restore_indices.data_ptr<int64_t>();
        for (int i = 0; i < padded_total; ++i) ri_vec[i] = ri64[i];
    }
    const int64_t* ri_ptr = ri_vec.data();

    auto* kv_restore_out = cp_kv_restore_unpad_indices_h_.data_ptr<int64_t>();
    auto* global_out     = cp_total_global_ids_h_.data_ptr<int64_t>();
    auto* local_out      = cp_total_local_ids_h_.data_ptr<int64_t>();
    auto* cu_kv_out      = cp_cu_kv_seqlens_global_h_.data_ptr<int32_t>();

    // Step 1: kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]
    int kv_restore_count = 0;
    for (int i = 0; i < padded_total; ++i) {
        if (pm_ptr[i] == 1) {
            kv_restore_out[kv_restore_count++] = ri_ptr[i];
        }
    }

    // Step 2: Build inverse permutation (inv_restore[kv_restore_indices[i]] = i)
    std::vector<int64_t> inv_restore(padded_total, -1);
    for (int i = 0; i < padded_total; ++i) {
        int64_t dst = ri_ptr[i];
        if (dst >= 0 && dst < padded_total) {
            inv_restore[dst] = i;
        }
    }

    // Step 3: Build pad_to_unpad cumsum (pad_to_unpad[i] = cumsum(padding_mask, 0..i) - 1)
    std::vector<int64_t> pad_to_unpad(padded_total);
    int64_t cumsum = 0;
    for (int i = 0; i < padded_total; ++i) {
        cumsum += pm_ptr[i];
        pad_to_unpad[i] = cumsum - 1;
    }

    // Step 4: Process q0_idx and q1_idx
    int total_ids_count = 0;

    auto process_q_indices = [&](const std::vector<int64_t>& q_idx) {
        for (size_t j = 0; j < q_idx.size(); ++j) {
            int64_t idx = q_idx[j];
            int64_t source_flat = (int64_t)cp_rank * local_tokens + idx;
            if (source_flat < 0 || source_flat >= padded_total) continue;
            int64_t global_padded = inv_restore[source_flat];
            if (global_padded < 0 || global_padded >= padded_total) continue;
            if (pm_ptr[global_padded] != 1) continue;
            int64_t global_unpadded = pad_to_unpad[global_padded];
            local_out[total_ids_count]  = idx;
            global_out[total_ids_count] = global_unpadded;
            total_ids_count++;
        }
    };

    process_q_indices(q0_idx);
    process_q_indices(q1_idx);

    if (total_ids_count == 0 && !q0_idx.empty()) {
        RTP_LLM_LOG_WARNING(
            "[SparseMlaParams::fillCpPlanParams] All q indices were filtered out "
            "(total_ids_count=0, q0_idx.size=%zu, q1_idx.size=%zu, "
            "cp_rank=%d, local_tokens=%d, padded_total=%d). "
            "This may indicate a data mismatch between CP chunking and padding mask.",
            q0_idx.size(), q1_idx.size(), cp_rank, local_tokens, padded_total);
    }

    // Step 5: cu_kv_seqlens_global = [0, cumsum(actual_input_lengths + prefix_lengths)]
    TORCH_CHECK(actual_input_lengths.scalar_type() == torch::kInt32,
                "fillCpPlanParams: actual_input_lengths must be int32, got ",
                actual_input_lengths.scalar_type());
    TORCH_CHECK(prefix_lengths.scalar_type() == torch::kInt32,
                "fillCpPlanParams: prefix_lengths must be int32, got ",
                prefix_lengths.scalar_type());
    const auto* ail_ptr = actual_input_lengths.data_ptr<int32_t>();
    const auto* pl_ptr  = prefix_lengths.data_ptr<int32_t>();
    cu_kv_out[0] = 0;
    for (int i = 0; i < batch_size; ++i) {
        cu_kv_out[i + 1] = cu_kv_out[i] + ail_ptr[i] + pl_ptr[i];
    }
    cp_total_kv_len = cu_kv_out[batch_size];

    // Step 6: Single cudaMemcpyAsync to device + set sizes
    refreshCpBuffer(kv_restore_count, total_ids_count, batch_size);
}

void SparseMlaParams::fillParams(torch_ext::PyAttentionInputs attn_inputs,
                                 int                          seq_size_per_block,
                                 bool                         forbid_realloc) {
    // Step 1: Call base class fillParams to fill shared parameters
    FlashInferMlaAttnParams::fillParams(attn_inputs.prefix_lengths,
                                        attn_inputs.sequence_lengths,
                                        attn_inputs.input_lengths,
                                        attn_inputs.kv_cache_kernel_block_id_host,
                                        seq_size_per_block,
                                        forbid_realloc);

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
            ensureTensorSize(batch_size, static_cast<int>(total_tokens), forbid_realloc);

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
            ensureTensorSize(batch_size, batch_size, forbid_realloc);

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
            [](rtp_llm::SparseMlaParams&    self,
               torch_ext::PyAttentionInputs attn_inputs,
               int                          seq_size_per_block,
               bool forbid_realloc) { self.fillParams(attn_inputs, seq_size_per_block, forbid_realloc); },
            pybind11::arg("attention_inputs"),
            pybind11::arg("seq_size_per_block"),
            pybind11::arg("forbid_realloc") = false)
        .def_readonly("expanded_seq_lens", &SparseMlaParams::expanded_seq_lens)
        .def_readonly("topk_indices_offset", &SparseMlaParams::topk_indices_offset)
        .def_readonly("ks", &SparseMlaParams::ks)
        .def_readonly("ke", &SparseMlaParams::ke)
        .def_readwrite("schedule_metadata", &SparseMlaParams::schedule_metadata)
        .def("fill_cp_plan_params",
             [](rtp_llm::SparseMlaParams&   self,
                const torch::Tensor&         padding_mask,
                const torch::Tensor&         kv_restore_indices,
                const std::vector<int64_t>&  q0_idx,
                const std::vector<int64_t>&  q1_idx,
                int                          cp_rank,
                int                          local_tokens,
                const torch::Tensor&         actual_input_lengths,
                const torch::Tensor&         prefix_lengths) {
                 self.fillCpPlanParams(padding_mask, kv_restore_indices,
                                       q0_idx, q1_idx, cp_rank, local_tokens,
                                       actual_input_lengths, prefix_lengths);
             },
             pybind11::arg("padding_mask"),
             pybind11::arg("kv_restore_indices"),
             pybind11::arg("q0_idx"),
             pybind11::arg("q1_idx"),
             pybind11::arg("cp_rank"),
             pybind11::arg("local_tokens"),
             pybind11::arg("actual_input_lengths"),
             pybind11::arg("prefix_lengths"))
        .def_readonly("cp_kv_restore_unpad_indices", &SparseMlaParams::cp_kv_restore_unpad_indices)
        .def_readonly("cp_total_global_ids", &SparseMlaParams::cp_total_global_ids)
        .def_readonly("cp_total_local_ids", &SparseMlaParams::cp_total_local_ids)
        .def_readonly("cp_cu_kv_seqlens_global", &SparseMlaParams::cp_cu_kv_seqlens_global)
        .def_readonly("cp_total_kv_len", &SparseMlaParams::cp_total_kv_len);

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
