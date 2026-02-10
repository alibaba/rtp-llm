#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
using namespace torch_ext;

namespace rtp_llm {
static const int                                      MIN_CACHE_PAGE_NUM        = 1024 * 1024;
static const int                                      MIN_CACHE_BATCH_SIZE      = 256;
static const int                                      MIN_CACHE_INPUT_TOKEN_NUM = 512;
std::tuple<torch::Tensor, std::vector<torch::Tensor>> FlashInferMlaAttnParams::allocateManyBuffer(
    const std::vector<std::vector<int64_t>>& shapes, bool is_device, torch::ScalarType dtype) {
    std::vector<torch::Tensor> tensors;
    std::vector<size_t>        sizes;
    size_t                     total_size = 0;

    // Calculate aligned sizes for each tensor
    for (const auto& shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        // Align to 32 elements for better memory access
        size = (size + 31) / 32 * 32;
        sizes.push_back(size);
        total_size += size;
    }

    // Allocate large continuous buffer
    torch::Tensor        buf;
    torch::TensorOptions options;
    if (is_device) {
        options = torch::dtype(dtype).device(torch::kCUDA).requires_grad(false);
        buf     = torch::empty({static_cast<int64_t>(total_size)}, options);
    } else {
        options = torch::dtype(dtype).device(torch::kCPU).requires_grad(false).pinned_memory(true);
        buf     = torch::empty({static_cast<int64_t>(total_size)}, options);
    }

    // Create tensor views using from_blob
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        // Calculate actual element count for this tensor
        size_t actual_size = 1;
        for (const auto dim : shapes[i]) {
            actual_size *= dim;
        }
        // Slice from buf and reshape to target shape
        tensors.emplace_back(buf.slice(0, offset, offset + actual_size).reshape(shapes[i]));
        offset += sizes[i];
    }

    return {buf, tensors};
}

void FlashInferMlaAttnParams::ensureTensorSize(int  batch_size,
                                               int  input_token_num,
                                               int  page_num,
                                               int  reuse_page_num,
                                               int  batch_reuse_info_size,
                                               bool is_cuda_graph,
                                               bool is_capture) {
    // Save old values for error reporting
    int    old_max_batch_size       = max_batch_size_;
    int    old_max_input_token_num  = max_input_token_num_;
    int    old_max_page_num         = max_page_num_;
    int    old_max_reuse_page_num   = max_reuse_page_num_;
    int    old_max_batch_reuse_info = max_batch_reuse_info_;
    size_t old_max_i64_elements     = max_i64_elements_;

    // Calculate required int64 elements for slot_mapping (aligned to 32)
    size_t required_i64_elements =
        (static_cast<size_t>(std::max(input_token_num, MIN_CACHE_INPUT_TOKEN_NUM)) + 31) / 32 * 32;

    // Check if we need to reallocate tensors
    bool need_realloc = (batch_size > max_batch_size_) || (input_token_num > max_input_token_num_)
                        || (page_num > max_page_num_) || (reuse_page_num > max_reuse_page_num_)
                        || (batch_reuse_info_size > max_batch_reuse_info_)
                        || (required_i64_elements > max_i64_elements_);

    // Check if reallocation is needed in CUDA graph replay mode (not capture mode)
    // During capture phase, reallocation is allowed to set up the graph
    if (need_realloc && is_cuda_graph && !is_capture) {
        RTP_LLM_LOG_ERROR(
            "[FlashInferMlaParams] Buffer reallocation required in CUDA graph mode, which is not allowed!");
        RTP_LLM_LOG_ERROR("Reallocation reason:");
        RTP_LLM_LOG_ERROR("Parameter changes:");
        if (batch_size > old_max_batch_size) {
            RTP_LLM_LOG_ERROR("  - max_batch_size: %d -> %d (requested: %d)",
                              old_max_batch_size,
                              std::max(MIN_CACHE_BATCH_SIZE, batch_size),
                              batch_size);
        }
        if (input_token_num > old_max_input_token_num) {
            RTP_LLM_LOG_ERROR("  - max_input_token_num: %d -> %d (requested: %d)",
                              old_max_input_token_num,
                              std::max(MIN_CACHE_INPUT_TOKEN_NUM, input_token_num),
                              input_token_num);
        }
        if (page_num > old_max_page_num) {
            RTP_LLM_LOG_ERROR("  - max_page_num: %d -> %d (requested: %d)",
                              old_max_page_num,
                              std::max(MIN_CACHE_PAGE_NUM, page_num),
                              page_num);
        }
        if (reuse_page_num > old_max_reuse_page_num) {
            RTP_LLM_LOG_ERROR("  - max_reuse_page_num: %d -> %d (requested: %d)",
                              old_max_reuse_page_num,
                              reuse_page_num,
                              reuse_page_num);
        }
        if (batch_reuse_info_size > old_max_batch_reuse_info) {
            RTP_LLM_LOG_ERROR("  - max_batch_reuse_info: %d -> %d (requested: %d)",
                              old_max_batch_reuse_info,
                              batch_reuse_info_size,
                              batch_reuse_info_size);
        }
        if (required_i64_elements > old_max_i64_elements) {
            RTP_LLM_LOG_ERROR("  - max_i64_elements (slot_mapping): %zu -> %zu (requested: %zu)",
                              old_max_i64_elements,
                              required_i64_elements,
                              required_i64_elements);
        }
        throw std::runtime_error("[FlashInferMlaParams] Buffer reallocation required in CUDA graph replay mode");
    }

    if (!need_realloc && buf_h.defined() && buf_d.defined() && buf_h_i64_.defined() && buf_d_i64_.defined()) {
        return;
    }

    // Update max sizes
    max_batch_size_       = std::max(max_batch_size_, batch_size);
    max_input_token_num_  = std::max(max_input_token_num_, input_token_num);
    max_page_num_         = std::max(max_page_num_, page_num);
    max_reuse_page_num_   = std::max(max_reuse_page_num_, reuse_page_num);
    max_batch_reuse_info_ = std::max(max_batch_reuse_info_, batch_reuse_info_size);
    max_i64_elements_     = required_i64_elements;

    max_batch_size_      = std::max(MIN_CACHE_BATCH_SIZE, max_batch_size_);
    max_input_token_num_ = std::max(MIN_CACHE_INPUT_TOKEN_NUM, max_input_token_num_);
    max_page_num_        = std::max(MIN_CACHE_PAGE_NUM, max_page_num_);

    // Allocate HOST buffer with all tensors in continuous memory
    auto alloc_ret_h = allocateManyBuffer({{max_input_token_num_},  // batch_indice
                                           {max_page_num_},         // page_indice
                                           {max_reuse_page_num_},   // reuse_cache_page_indice
                                           {max_batch_size_ + 1},   // decode_page_indptr
                                           {max_batch_size_ + 1},   // prefill_page_indptr
                                           {max_batch_size_},       // paged_kv_last_page_len
                                           {max_batch_size_ + 1},   // qo_indptr
                                           {max_batch_size_},       // kvlen
                                           {max_input_token_num_},  // positions
                                           {max_batch_size_, 4}},   // batch_reuse_info_vec (2D: [batch_size, 4])
                                          false);

    buf_h                     = std::get<0>(alloc_ret_h);
    auto& tensors_h           = std::get<1>(alloc_ret_h);
    batch_indice_h            = tensors_h[0];
    page_indice_h             = tensors_h[1];
    reuse_cache_page_indice_h = tensors_h[2];
    decode_page_indptr_h      = tensors_h[3];
    prefill_page_indptr_h     = tensors_h[4];
    paged_kv_last_page_len_h  = tensors_h[5];
    qo_indptr_h               = tensors_h[6];
    kvlen_h                   = tensors_h[7];
    positions_h               = tensors_h[8];
    batch_reuse_info_vec_h    = tensors_h[9];

    // Allocate DEVICE buffer with all tensors in continuous memory
    auto alloc_ret_d = allocateManyBuffer({{max_input_token_num_},  // batch_indice
                                           {max_page_num_},         // page_indice
                                           {max_reuse_page_num_},   // reuse_cache_page_indice
                                           {max_batch_size_ + 1},   // decode_page_indptr
                                           {max_batch_size_ + 1},   // prefill_page_indptr
                                           {max_batch_size_},       // paged_kv_last_page_len
                                           {max_batch_size_ + 1},   // qo_indptr
                                           {max_batch_size_},       // kvlen
                                           {max_input_token_num_},  // positions
                                           {max_batch_size_, 4}},   // batch_reuse_info_vec (2D: [batch_size, 4])
                                          true);

    buf_d                     = std::get<0>(alloc_ret_d);
    auto& tensors_d           = std::get<1>(alloc_ret_d);
    batch_indice_d            = tensors_d[0];
    page_indice_d             = tensors_d[1];
    reuse_cache_page_indice_d = tensors_d[2];
    decode_page_indptr_d      = tensors_d[3];
    prefill_page_indptr_d     = tensors_d[4];
    paged_kv_last_page_len_d  = tensors_d[5];
    qo_indptr_d               = tensors_d[6];
    kvlen_d                   = tensors_d[7];
    positions_d               = tensors_d[8];
    batch_reuse_info_vec_d    = tensors_d[9];

    // Allocate int64 buffers for slot_mapping
    auto alloc_ret_i64_h = allocateManyBuffer({{static_cast<int64_t>(max_i64_elements_)}}, false, torch::kInt64);
    buf_h_i64_           = std::get<0>(alloc_ret_i64_h);
    slot_mapping_h_      = std::get<1>(alloc_ret_i64_h)[0];

    auto alloc_ret_i64_d = allocateManyBuffer({{static_cast<int64_t>(max_i64_elements_)}}, true, torch::kInt64);
    buf_d_i64_           = std::get<0>(alloc_ret_i64_d);
    slot_mapping_d_      = std::get<1>(alloc_ret_i64_d)[0];
}

void FlashInferMlaAttnParams::fillParamsInternal(torch::Tensor t_prefix_lengths,
                                                 torch::Tensor t_sequence_lengths,
                                                 torch::Tensor t_input_lengths,
                                                 torch::Tensor t_kv_cache_block_id_host,
                                                 int           batch_size,
                                                 int           seq_size_per_block,
                                                 int&          input_token_num,
                                                 int&          page_num,
                                                 int&          reuse_page_num,
                                                 int&          batch_reuse_info_size) {
    const int max_batch_blocks = t_kv_cache_block_id_host.defined() && t_kv_cache_block_id_host.size(0) > 0 ?
                                     t_kv_cache_block_id_host.size(1) :
                                     -1;

    RTP_LLM_CHECK_WITH_INFO(
        batch_size <= max_batch_size_, "batch_size exceed reserved %d > %d", batch_size, max_batch_size_);

    // Get pointers to HOST tensors
    auto batch_indice_ptr            = batch_indice_h.data_ptr<int32_t>();
    auto page_indice_ptr             = page_indice_h.data_ptr<int32_t>();
    auto reuse_cache_page_indice_ptr = reuse_cache_page_indice_h.data_ptr<int32_t>();
    auto decode_page_indptr_ptr      = decode_page_indptr_h.data_ptr<int32_t>();
    auto prefill_page_indptr_ptr     = prefill_page_indptr_h.data_ptr<int32_t>();
    auto paged_kv_last_page_len_ptr  = paged_kv_last_page_len_h.data_ptr<int32_t>();
    auto qo_indptr_ptr               = qo_indptr_h.data_ptr<int32_t>();
    auto kvlen_ptr                   = kvlen_h.data_ptr<int32_t>();
    auto positions_ptr               = positions_h.data_ptr<int32_t>();
    auto batch_reuse_info_vec_ptr    = batch_reuse_info_vec_h.data_ptr<int32_t>();

    // Get input data pointers
    auto input_lengths = t_input_lengths.data_ptr<int32_t>();
    auto prefix_lengths =
        t_prefix_lengths.defined() && t_prefix_lengths.size(0) > 0 ? t_prefix_lengths.data_ptr<int32_t>() : nullptr;
    auto sequence_lengths  = t_sequence_lengths.defined() && t_sequence_lengths.size(0) > 0 ?
                                 t_sequence_lengths.data_ptr<int32_t>() :
                                 nullptr;
    auto kv_cache_block_id = t_kv_cache_block_id_host.defined() && t_kv_cache_block_id_host.size(0) > 0 ?
                                 t_kv_cache_block_id_host.data_ptr<int32_t>() :
                                 nullptr;

    int max_kv_len      = 0;
    int max_q_len       = 0;
    int accu_q_len      = 0;
    int accu_kv_len     = 0;
    int offset          = 0;
    int total_page_idx  = 0;
    int reuse_page_idx  = 0;
    int batch_start_idx = 0;

    // Initialize indptr arrays
    decode_page_indptr_ptr[0]  = 0;
    prefill_page_indptr_ptr[0] = 0;
    qo_indptr_ptr[0]           = 0;

    for (int i = 0; i < batch_size; i++) {
        int seq_len = 0;
        if (prefix_lengths) {
            int input_length  = input_lengths[i];
            int prefix_length = prefix_lengths[i];

            RTP_LLM_CHECK_WITH_INFO(offset + input_length <= max_input_token_num_,
                                    "input_token_num exceed reserved %d > %d",
                                    offset + input_length,
                                    max_input_token_num_);

            for (int j = 0; j < input_length; j++) {
                batch_indice_ptr[offset] = i;
                positions_ptr[offset]    = j + prefix_length;
                offset += 1;
            }
            seq_len   = input_length + prefix_length;
            max_q_len = std::max(max_q_len, input_length);
            accu_q_len += input_length;
            accu_kv_len += seq_len;

            int reuse_page_num = (prefix_length + seq_size_per_block - 1) / seq_size_per_block;
            if (kv_cache_block_id) {
                RTP_LLM_CHECK_WITH_INFO(reuse_page_idx + reuse_page_num <= max_reuse_page_num_,
                                        "reuse_page_num exceed reserved %d > %d",
                                        reuse_page_idx + reuse_page_num,
                                        max_reuse_page_num_);
                for (int j = 0; j < reuse_page_num; j++) {
                    auto page_idx                                 = kv_cache_block_id[i * max_batch_blocks + j];
                    reuse_cache_page_indice_ptr[reuse_page_idx++] = page_idx;
                }
            }
            if (prefix_length) {
                RTP_LLM_CHECK_WITH_INFO(
                    i < max_batch_size_, "batch_index exceed reserved %d >= %d", i, max_batch_size_);
                // batch_reuse_info_vec is 2D: [batch_size, 4]
                batch_reuse_info_vec_ptr[i * 4]     = i;
                batch_reuse_info_vec_ptr[i * 4 + 1] = prefix_length;
                batch_reuse_info_vec_ptr[i * 4 + 2] = batch_start_idx;
                batch_reuse_info_vec_ptr[i * 4 + 3] = reuse_page_num;
                batch_start_idx += reuse_page_num;
            } else {
                RTP_LLM_CHECK_WITH_INFO(
                    i < max_batch_size_, "batch_index exceed reserved %d >= %d", i, max_batch_size_);
                // batch_reuse_info_vec is 2D: [batch_size, 4]
                batch_reuse_info_vec_ptr[i * 4]     = i;
                batch_reuse_info_vec_ptr[i * 4 + 1] = 0;
                batch_reuse_info_vec_ptr[i * 4 + 2] = 0;
                batch_reuse_info_vec_ptr[i * 4 + 3] = 0;
            }
        } else {
            // Decode mode: ensure batch_size <= max_input_token_num_
            RTP_LLM_CHECK_WITH_INFO(batch_size <= max_input_token_num_,
                                    "batch_size exceed max_input_token_num_ in decode mode %d > %d",
                                    batch_size,
                                    max_input_token_num_);
            batch_indice_ptr[i] = i;
            positions_ptr[i]    = sequence_lengths[i];
            seq_len             = sequence_lengths[i] + 1;
            accu_q_len += 1;
            accu_kv_len += 1;
        }
        paged_kv_last_page_len_ptr[i] = (seq_len - 1) % seq_size_per_block + 1;
        kvlen_ptr[i]                  = seq_len;
        max_kv_len                    = std::max(seq_len, max_kv_len);

        int current_page_num = (seq_len + seq_size_per_block - 1) / seq_size_per_block;
        RTP_LLM_CHECK_WITH_INFO(total_page_idx + current_page_num <= max_page_num_,
                                "page_num exceed reserved %d > %d",
                                total_page_idx + current_page_num,
                                max_page_num_);
        if (kv_cache_block_id) {
            for (int j = 0; j < current_page_num; j++) {
                auto page_idx                     = kv_cache_block_id[i * max_batch_blocks + j];
                page_indice_ptr[total_page_idx++] = page_idx;
            }
        }
        decode_page_indptr_ptr[i + 1]  = total_page_idx;
        prefill_page_indptr_ptr[i + 1] = accu_kv_len;
        qo_indptr_ptr[i + 1]           = accu_q_len;
    }

    input_token_num       = offset > 0 ? offset : batch_size;
    page_num              = total_page_idx;
    reuse_page_num        = reuse_page_idx;
    batch_reuse_info_size = batch_size * 4;  // 4 ints per batch entry
}

void FlashInferMlaAttnParams::refreshBuffer(
    int batch_size, int input_token_num, int page_num, int reuse_page_num, int batch_reuse_info_size) {
    // Get current CUDA stream
    cudaStream_t stream = GET_CURRENT_STREAM();

    // Single async copy from HOST to DEVICE for the entire buffer
    // Since all tensors are in continuous memory, we can copy the entire buffer at once
    size_t total_bytes = buf_h.numel() * sizeof(int32_t);
    cudaMemcpyAsync(buf_d.data_ptr(), buf_h.data_ptr(), total_bytes, cudaMemcpyHostToDevice, stream);

    // Update tensor shapes (without reallocating memory)
    // Use vector<int64_t> which can be implicitly converted to c10::IntArrayRef
    std::vector<int64_t> shape;

    // Update shapes for batch_size + 1 tensors
    shape = {batch_size + 1};
    decode_page_indptr_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    decode_page_indptr_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    prefill_page_indptr_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    prefill_page_indptr_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    qo_indptr_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    qo_indptr_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);

    // Update shapes for input_token_num tensors
    shape = {input_token_num};
    batch_indice_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    batch_indice_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    positions_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    positions_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);

    // Update shapes for batch_size tensors
    shape = {batch_size};
    kvlen_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    kvlen_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    paged_kv_last_page_len_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    paged_kv_last_page_len_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);

    // Update shapes for page_num tensors
    shape = {page_num};
    page_indice_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    page_indice_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);

    // Update shapes for reuse_page_num tensors
    shape = {reuse_page_num};
    reuse_cache_page_indice_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    reuse_cache_page_indice_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);

    // Update shape for batch_reuse_info_vec
    shape = {batch_size, 4};
    batch_reuse_info_vec_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
    batch_reuse_info_vec_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
}

void FlashInferMlaAttnParams::fillParams(torch::Tensor t_prefix_lengths,
                                         torch::Tensor t_sequence_lengths,
                                         torch::Tensor t_input_lengths,
                                         torch::Tensor t_kv_cache_block_id_host,
                                         int           seq_size_per_block,
                                         bool          is_cuda_graph,
                                         bool          is_capture) {
    const int batch_size = t_input_lengths.size(0);

    // First pass: calculate required sizes accurately
    auto input_lengths_ptr = t_input_lengths.data_ptr<int32_t>();
    auto prefix_lengths_ptr =
        t_prefix_lengths.defined() && t_prefix_lengths.size(0) > 0 ? t_prefix_lengths.data_ptr<int32_t>() : nullptr;
    auto sequence_lengths_ptr = t_sequence_lengths.defined() && t_sequence_lengths.size(0) > 0 ?
                                    t_sequence_lengths.data_ptr<int32_t>() :
                                    nullptr;

    int input_token_num       = 0;
    int page_num              = 0;
    int reuse_page_num        = 0;
    int batch_reuse_info_size = batch_size * 4;  // 4 ints per batch entry

    for (int i = 0; i < batch_size; i++) {
        int seq_len = 0;
        if (prefix_lengths_ptr) {
            int input_length  = input_lengths_ptr[i];
            int prefix_length = prefix_lengths_ptr[i];
            input_token_num += input_length;
            seq_len = input_length + prefix_length;
            reuse_page_num += (prefix_length + seq_size_per_block - 1) / seq_size_per_block;
        } else {
            input_token_num += 1;
            seq_len = sequence_lengths_ptr[i] + 1;
        }
        page_num += (seq_len + seq_size_per_block - 1) / seq_size_per_block;
    }

    // Ensure tensors are allocated with sufficient size
    ensureTensorSize(
        batch_size, input_token_num, page_num, reuse_page_num, batch_reuse_info_size, is_cuda_graph, is_capture);

    // Fill params directly into HOST tensors
    fillParamsInternal(t_prefix_lengths,
                       t_sequence_lengths,
                       t_input_lengths,
                       t_kv_cache_block_id_host,
                       batch_size,
                       seq_size_per_block,
                       input_token_num,
                       page_num,
                       reuse_page_num,
                       batch_reuse_info_size);

    // Refresh buffer (copy to DEVICE and update shapes)
    refreshBuffer(batch_size, input_token_num, page_num, reuse_page_num, batch_reuse_info_size);

    batch_indice            = batch_indice_d;
    page_indice             = page_indice_d;
    reuse_cache_page_indice = reuse_page_num > 0 ? reuse_cache_page_indice_d : torch::Tensor();
    decode_page_indptr      = decode_page_indptr_d;
    prefill_page_indptr     = prefill_page_indptr_d;
    paged_kv_last_page_len  = paged_kv_last_page_len_d;
    qo_indptr               = qo_indptr_d;
    kvlen                   = kvlen_d;
    positions               = positions_d;
    batch_reuse_info_vec    = batch_size > 0 ? batch_reuse_info_vec_d : torch::Tensor();

    // Calculate slot_mapping
    if (t_kv_cache_block_id_host.defined() && t_kv_cache_block_id_host.numel() > 0 && input_token_num > 0) {
        const int64_t max_blocks       = t_kv_cache_block_id_host.size(1);
        auto          block_table_ptr  = t_kv_cache_block_id_host.data_ptr<int32_t>();
        auto          batch_indice_ptr = batch_indice_h.data_ptr<int32_t>();
        auto          positions_ptr    = positions_h.data_ptr<int32_t>();

        slot_mapping_h_       = buf_h_i64_.slice(0, 0, input_token_num).reshape({input_token_num});
        auto slot_mapping_ptr = slot_mapping_h_.data_ptr<int64_t>();

        for (int64_t i = 0; i < input_token_num; ++i) {
            const int32_t batch_id     = batch_indice_ptr[i];
            const int32_t position     = positions_ptr[i];
            const int32_t block_index  = position / seq_size_per_block;
            const int32_t block_offset = position % seq_size_per_block;
            const int32_t block_number = block_table_ptr[batch_id * max_blocks + block_index];
            slot_mapping_ptr[i]        = static_cast<int64_t>(block_number) * seq_size_per_block + block_offset;
        }

        cudaStream_t stream      = GET_CURRENT_STREAM();
        size_t       total_bytes = static_cast<size_t>(input_token_num) * sizeof(int64_t);
        cudaMemcpyAsync(
            slot_mapping_d_.data_ptr(), slot_mapping_h_.data_ptr(), total_bytes, cudaMemcpyHostToDevice, stream);

        slot_mapping_d_.unsafeGetTensorImpl()->set_sizes_contiguous({input_token_num});
        slot_mapping = slot_mapping_d_;
    } else {
        slot_mapping = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    }

    return;
}

void registerPyFlashInferMlaParams(pybind11::module& m) {
    pybind11::class_<FlashInferMlaAttnParams, std::shared_ptr<FlashInferMlaAttnParams>, rtp_llm::ParamsBase>(
        m, "FlashInferMlaAttnParams")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::FlashInferMlaAttnParams& self,
               torch::Tensor                     prefix_lengths,
               torch::Tensor                     sequence_lengths,
               torch::Tensor                     input_lengths,
               torch::Tensor                     kv_cache_block_id_host,
               int                               seq_size_per_block,
               bool                              is_cuda_graph,
               bool                              is_capture) {
                self.fillParams(prefix_lengths,
                                sequence_lengths,
                                input_lengths,
                                kv_cache_block_id_host,
                                seq_size_per_block,
                                is_cuda_graph,
                                is_capture);
            },
            pybind11::arg("prefix_lengths"),
            pybind11::arg("sequence_lengths"),
            pybind11::arg("input_lengths"),
            pybind11::arg("kv_cache_block_id_host"),
            pybind11::arg("seq_size_per_block"),
            pybind11::arg("is_cuda_graph") = false,
            pybind11::arg("is_capture")    = false,
            "Fill parameters for CUDA graph execution")
        // HOST tensors (_h suffix)
        .def_readonly("batch_indice_h", &FlashInferMlaAttnParams::batch_indice_h, "Batch indices on HOST")
        .def_readonly("page_indice_h", &FlashInferMlaAttnParams::page_indice_h, "Page indices on HOST")
        .def_readonly("reuse_cache_page_indice_h",
                      &FlashInferMlaAttnParams::reuse_cache_page_indice_h,
                      "Reuse cache page indices on HOST")
        .def_readonly(
            "decode_page_indptr_h", &FlashInferMlaAttnParams::decode_page_indptr_h, "Decode page indptr on HOST")
        .def_readonly(
            "prefill_page_indptr_h", &FlashInferMlaAttnParams::prefill_page_indptr_h, "Prefill page indptr on HOST")
        .def_readonly("paged_kv_last_page_len_h",
                      &FlashInferMlaAttnParams::paged_kv_last_page_len_h,
                      "Paged KV last page length on HOST")
        .def_readonly("qo_indptr_h", &FlashInferMlaAttnParams::qo_indptr_h, "Query/output indptr on HOST")
        .def_readonly("kvlen_h", &FlashInferMlaAttnParams::kvlen_h, "KV length on HOST")
        .def_readonly("positions_h", &FlashInferMlaAttnParams::positions_h, "Positions on HOST")
        .def_readonly("batch_reuse_info_vec_h",
                      &FlashInferMlaAttnParams::batch_reuse_info_vec_h,
                      "Batch reuse info vector on HOST")
        // DEVICE tensors (_d suffix)
        .def_readonly("batch_indice_d", &FlashInferMlaAttnParams::batch_indice_d, "Batch indices on DEVICE")
        .def_readonly("page_indice_d", &FlashInferMlaAttnParams::page_indice_d, "Page indices on DEVICE")
        .def_readonly("reuse_cache_page_indice_d",
                      &FlashInferMlaAttnParams::reuse_cache_page_indice_d,
                      "Reuse cache page indices on DEVICE")
        .def_readonly(
            "decode_page_indptr_d", &FlashInferMlaAttnParams::decode_page_indptr_d, "Decode page indptr on DEVICE")
        .def_readonly(
            "prefill_page_indptr_d", &FlashInferMlaAttnParams::prefill_page_indptr_d, "Prefill page indptr on DEVICE")
        .def_readonly("paged_kv_last_page_len_d",
                      &FlashInferMlaAttnParams::paged_kv_last_page_len_d,
                      "Paged KV last page length on DEVICE")
        .def_readonly("qo_indptr_d", &FlashInferMlaAttnParams::qo_indptr_d, "Query/output indptr on DEVICE")
        .def_readonly("kvlen_d", &FlashInferMlaAttnParams::kvlen_d, "KV length on DEVICE")
        .def_readonly("positions_d", &FlashInferMlaAttnParams::positions_d, "Positions on DEVICE")
        .def_readonly("batch_reuse_info_vec_d",
                      &FlashInferMlaAttnParams::batch_reuse_info_vec_d,
                      "Batch reuse info vector on DEVICE")
        // slot_mapping output
        .def_readonly("slot_mapping", &FlashInferMlaAttnParams::slot_mapping, "Slot mapping for KV cache");
    m.def(
        "fill_mla_params",
        [](torch::Tensor t_prefill_lengths,
           torch::Tensor t_sequence_lengths,
           torch::Tensor t_input_lengths,
           torch::Tensor t_kv_cache_block_id_host,
           int           seq_size_per_block) {
            auto params = std::make_shared<rtp_llm::FlashInferMlaAttnParams>();
            params->fillParams(
                t_prefill_lengths, t_sequence_lengths, t_input_lengths, t_kv_cache_block_id_host, seq_size_per_block);
            return params;
        },
        pybind11::arg("t_prefill_lengths"),
        pybind11::arg("t_sequence_lengths"),
        pybind11::arg("t_input_lengths"),
        pybind11::arg("t_kv_cache_block_id_host"),
        pybind11::arg("seq_size_per_block"));
}

}  // namespace rtp_llm