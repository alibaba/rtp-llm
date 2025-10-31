
#include <iostream>
#include <numeric>
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "flashmla/flashmla.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#endif

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

using Slice          = torch::indexing::Slice;
constexpr auto TNone = torch::indexing::None;

static const int MIN_CACHE_BATCH_SIZE      = 256;
static const int MIN_CACHE_INPUT_TOKEN_NUM = 512;
static const int MIN_CACHE_PAGE_NUM        = 128 * 1024;

bool FlashInferAttnParams::isDecode(int input_token_num) {
    return input_token_num <= MIN_CACHE_INPUT_TOKEN_NUM * 2;
}

void FlashInferAttnParams::recycle(void* p) {
    auto flashinfer = (FlashInferAttnParams*)p;
    if (isDecode(flashinfer->input_token_num)) {
        ParamsCache::DECODE_PARAMS_CACHE.push_back(flashinfer);
    } else {
        ParamsCache::PREFILL_PARAMS_CACHE.push_back(flashinfer);
    }
}

bool FlashInferAttnParams::check_recycle() {
    return true;
}

FlashInferAttnParams* FlashInferAttnParams::get(int batch_size, int input_token_num) {
    auto cache = isDecode(input_token_num) ? &ParamsCache::DECODE_PARAMS_CACHE : &ParamsCache::PREFILL_PARAMS_CACHE;
    if (!cache->empty()) {
        auto params = cache->back();
        cache->pop_back();
        if (batch_size <= params->batch_size && input_token_num <= params->input_token_num) {
            return params;
        }
        delete params;
    }
    return nullptr;
}

tuple<BufferPtr, vector<torch::Tensor>> FlashInferAttnParams::allocateManyBuffer(
    CudaDevice* device, const std::vector<std::vector<int64_t>>& shapes, AllocationType atype) {
    vector<torch::Tensor> tensors;
    vector<size_t>        sizes;
    size_t                total_size = 0;
    for (const auto& shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        size = (size + 31) / 32 * 32;
        sizes.push_back(size);
        total_size += size;
    }
    auto buf         = device->allocateBuffer({DataType::TYPE_INT32, {total_size}, atype}, {"flashinfer_buf"});
    auto buf_ptr     = buf->data<int>();
    auto cuda_option = torch::dtype(torch::kInt).device(torch::DeviceType::CUDA).requires_grad(false);

    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        tensors.emplace_back(torch::from_blob(buf_ptr + offset, shapes[i], cuda_option));
        offset += sizes[i];
    }
    return {buf, tensors};
}

FlashInferAttnParams*
FlashInferAttnParams::create(CudaDevice* device, int batch_size, int input_token_num, int page_num) {
    if (auto params = get(batch_size, input_token_num)) {
        return params;
    }
    RTP_LLM_LOG_DEBUG("new FlashInferAttnParams batch_size(%d) input_token_num(%d)", batch_size, input_token_num);
    auto params             = make_unique<FlashInferAttnParams>();
    params->batch_size      = batch_size;
    params->input_token_num = input_token_num;
    params->page_num        = page_num;

    // batch_prefill_tmp_v may use 256M buffer
    params->float_workspace = device->allocateBuffer(
        {DataType::TYPE_INT8, {(256 + 16) * 1024 * 1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::HOST}, {"int_host_workspace"});

    params->float_workspace_d = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_d   = Buffer2torchTensor(params->int_workspace, false);
    params->int_workspace_h   = Buffer2torchTensor(params->int_host_workspace, false);

#define ALLOC_BUFFER(suffix, type)                                                                                     \
    do {                                                                                                               \
        auto alloc_ret = allocateManyBuffer(device,                                                                    \
                                            {{batch_size + 1},  /* page_indptr */                                      \
                                             {batch_size + 1},  /* qo_indptr */                                        \
                                             {input_token_num}, /* batch_indice */                                     \
                                             {input_token_num}, /* positions */                                        \
                                             {batch_size},      /* kv_len */                                           \
                                             {batch_size},      /* paged_kv_last_page_len */                           \
                                             {page_num}},       /* page_indice */                                      \
                                            type);                                                                     \
                                                                                                                       \
        params->buf_##suffix                    = std::get<0>(alloc_ret);                                              \
        auto& tensors                           = std::get<1>(alloc_ret);                                              \
        params->page_indptr_##suffix            = tensors[0];                                                          \
        params->qo_indptr_##suffix              = tensors[1];                                                          \
        params->batch_indice_##suffix           = tensors[2];                                                          \
        params->positions_##suffix              = tensors[3];                                                          \
        params->kvlen_##suffix                  = tensors[4];                                                          \
        params->paged_kv_last_page_len_##suffix = tensors[5];                                                          \
        params->page_indice_##suffix            = tensors[6];                                                          \
    } while (0)

    ALLOC_BUFFER(h, AllocationType::HOST);
    ALLOC_BUFFER(d, AllocationType::DEVICE);

    return params.release();
}

void FlashInferAttnParams::fillParams(torch::Tensor sequence_lengths,
                                      torch::Tensor input_lengths,
                                      torch::Tensor kv_cache_block_id_host,
                                      int           batch_size,
                                      int           seq_size_per_block) {
    fillFlashInfer(nullptr,
                   torchTensor2Buffer(sequence_lengths),
                   torchTensor2Buffer(input_lengths),
                   torchTensor2Buffer(kv_cache_block_id_host),
                   batch_size,
                   seq_size_per_block);
    refreshFlashInferBuf(
        dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice()), batch_size, input_lengths.size(0));
}

void FlashInferAttnParams::fillFlashInfer(const BufferPtr& prefix_lengths_host,
                                          const BufferPtr& sequence_lengths_host,
                                          const BufferPtr& input_lengths_host,
                                          const BufferPtr& kv_cache_block_id_host,
                                          const int        batch_size,
                                          const int        tokens_per_block) {
    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    RTP_LLM_CHECK_WITH_INFO(
        batch_size <= this->batch_size, "batch_size exceed reserved %d > %d", batch_size, this->batch_size);
    auto qo_indptr              = qo_indptr_h.data_ptr<int>();
    auto page_indptr            = page_indptr_h.data_ptr<int>();
    auto batch_indice           = batch_indice_h.data_ptr<int>();
    auto positions              = positions_h.data_ptr<int>();
    auto paged_kv_last_page_len = paged_kv_last_page_len_h.data_ptr<int>();
    auto kvlen                  = kvlen_h.data_ptr<int>();
    auto page_indice            = page_indice_h.data_ptr<int>();
    auto input_lengths          = input_lengths_host->data<int>();
    auto prefix_lengths         = prefix_lengths_host ? prefix_lengths_host->data<int>() : nullptr;
    auto sequence_lengths       = sequence_lengths_host ? sequence_lengths_host->data<int>() : nullptr;
    auto kv_cache_block_id      = kv_cache_block_id_host ? kv_cache_block_id_host->data<int>() : nullptr;
    int  offset                 = 0;
    int  total_page_idx         = 0;
    qo_indptr[0]                = 0;
    page_indptr[0]              = 0;
    max_q_len                   = 1;
    accu_q_len                  = 0;
    for (int i = 0; i < batch_size; i++) {
        int seq_len = 0;
        if (prefix_lengths) {
            int input_length  = input_lengths[i];
            int prefix_length = prefix_lengths[i];
            RTP_LLM_CHECK_WITH_INFO(offset + input_length <= this->input_token_num,
                                    "token_num exceed reserved %d > %d",
                                    offset + input_length,
                                    this->input_token_num);
            for (int j = 0; j < input_length; j++) {
                batch_indice[offset] = i;
                positions[offset]    = j + prefix_length;
                offset += 1;
            }
            seq_len   = input_length + prefix_length;
            max_q_len = max(max_q_len, input_length);
            accu_q_len += input_length;
        } else {
            batch_indice[i] = i;
            positions[i]    = sequence_lengths[i];
            seq_len         = sequence_lengths[i] + 1;
            accu_q_len += 1;
        }
        paged_kv_last_page_len[i] = (seq_len - 1) % tokens_per_block + 1;
        kvlen[i]                  = seq_len;
        max_kv_len                = max(seq_len, max_kv_len);

        int page_num = (seq_len + tokens_per_block - 1) / tokens_per_block;
        RTP_LLM_CHECK_WITH_INFO(total_page_idx + page_num <= this->page_num,
                                "page_num exceed reserved %d > %d",
                                total_page_idx + page_num,
                                this->page_num);
        if (kv_cache_block_id) {
            for (int j = 0; j < page_num; j++) {
                auto page_idx                 = kv_cache_block_id[i * max_batch_blocks + j];
                page_indice[total_page_idx++] = page_idx;
            }
        }
        page_indptr[i + 1] = total_page_idx;
        qo_indptr[i + 1]   = accu_q_len;
    }
}

void FlashInferAttnParams::refreshFlashInferBuf(CudaDevice* device, int batch_size, int input_token_num) {
    auto stream = device->getStream();
    cudaMemcpyAsync(buf_d->data(), buf_h->data(), buf_h->sizeBytes(), cudaMemcpyHostToDevice, stream);

    vector<int64_t> shape = {batch_size + 1};
#define REFRESH_SHAPE(t)                                                                                               \
    do {                                                                                                               \
        t##_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);                                                      \
        t##_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);                                                      \
    } while (0)

    REFRESH_SHAPE(page_indptr);
    REFRESH_SHAPE(qo_indptr);

    shape[0] = input_token_num;
    REFRESH_SHAPE(batch_indice);
    REFRESH_SHAPE(positions);

    shape[0] = batch_size;
    REFRESH_SHAPE(kvlen);
    REFRESH_SHAPE(paged_kv_last_page_len);
}

bool FlashInferAttnParams::sameQLength(const BufferPtr& input_lengths_host, int batch_size, int& q_length) {
    auto input_lengths = input_lengths_host->data<int>();
    for (int i = 0; i < batch_size; i++) {
        int input_length = input_lengths[i];
        if (q_length > 0 && q_length != input_length) {
            return false;
        }
        q_length = input_length;
    }
    return true;
}

void FlashInferAttnParams::genPlan(int     batch_size,
                                   int     q_length,
                                   int     local_head_num,
                                   int     local_head_num_kv,
                                   int     size_per_head,
                                   int     tokens_per_block,
                                   int     kv_lora_rank,
                                   bool    use_mla,
                                   int64_t stream,
                                   bool    enable_cuda_graph) {
    if (use_mla) {
        if (mla_ops_type == MlaOpsType::FLASH_INFER) {
            plan = BatchMLAPagedAttentionPlan(float_workspace_d,
                                              int_workspace_d,
                                              int_workspace_h,
                                              qo_indptr_h,
                                              page_indptr_h,
                                              kvlen_h,
                                              local_head_num,
                                              kv_lora_rank,
                                              true,
                                              stream);
        } else if (mla_ops_type == MlaOpsType::FLASH_MLA) {
            flash_mla_plan = get_mla_metadata(kvlen_d, local_head_num * q_length, 1);
        } else {
            RTP_LLM_FAIL("unexpected mla ops type: %d", int(mla_ops_type));
        }
    } else {
        if (decode_plan) {
            plan = BatchDecodeWithPagedKVCachePlan(
                float_workspace_d,  // float_workspace_buffer
                int_workspace_d,    // int_workspace_buffer
                int_workspace_h,    // page_locked_int_workspace_buffer
                page_indptr_h,      // indptr
                batch_size,         // batch_size
                local_head_num,     // num_qo_heads
                local_head_num_kv,  // num_kv_heads
                tokens_per_block,   // page_size
                enable_cuda_graph,  // enable_cuda_graph,
                -1,                 // window_left
                -1,                 // logits_soft_cap
                size_per_head,      // head_dim_qk
                size_per_head,      // head_dim_vo
                torch::empty(0, dataTypeToTorchType(dtype == DataType::TYPE_FP8_E4M3 ? DataType::TYPE_FP16 : dtype)),
                torch::empty(0, dataTypeToTorchType(dtype)),  // empty_kv_data
                stream);

        } else {
            plan = BatchPrefillWithKVCachePlan(
                float_workspace_d,                                           // float_workspace_buffer
                int_workspace_d,                                             // int_workspace_buffer
                int_workspace_h,                                             // page_locked_int_workspace_buffer
                qo_indptr_h,                                                 // qo_indptr
                page_indptr_h,                                               // kv_indptr
                torch::empty(0, dataTypeToTorchType(DataType::TYPE_INT32)),  // kv_len_arr, not in use yet
                accu_q_len,                                                  // total_num_rows
                batch_size,                                                  // batch_size
                local_head_num,                                              // num_qo_heads
                local_head_num_kv,                                           // num_kv_heads
                tokens_per_block,                                            // page_size
                false,                                                       // enable_cuda_graph
                size_per_head,                                               // head_dim_qk
                size_per_head,                                               // head_dim_vo
                true,                                                        // causal
                stream);
        }
    }
}

bool FlashInferAttnParams::check(rtp_llm::DeviceBase*             device,
                                 const rtp_llm::AttentionConfigs& attn_configs,
                                 DataType                         dtype,
                                 bool                             is_prefill) {
    if (rtp_llm::get_sm() < 80) {
        return false;
    }
    auto cuda_device = dynamic_cast<CudaDevice*>(device);
    if (!cuda_device) {
        return false;
    }
    const bool disable_flash_infer = device->initParams().fmha_config.disable_flash_infer;
    MlaOpsType mla_ops_type        = device->mla_ops_type;
    if ((!attn_configs.use_mla || mla_ops_type == MlaOpsType::FLASH_INFER) && disable_flash_infer) {
        return false;
    }
    if (!attn_configs.use_mla) {
        const int size_per_head = attn_configs.size_per_head;
        const int group_size    = attn_configs.head_num / attn_configs.kv_head_num;
        if ((dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16 && dtype != DataType::TYPE_FP8_E4M3)
            || (attn_configs.kv_cache_dtype != KvCacheDataType::BASE
                && attn_configs.kv_cache_dtype != KvCacheDataType::FP8)
            || (attn_configs.rope_config.style != RopeStyle::Base && attn_configs.rope_config.style != RopeStyle::No)
            || attn_configs.mask_type != causalMask || attn_configs.q_scaling != 1.0f || attn_configs.use_logn_attn
            || (size_per_head != 64 && size_per_head != 128 && size_per_head != 192)
            || (!is_prefill && group_size > 10 && group_size != 16 && group_size != 12)) {
            return false;
        }
    }
    return true;
}

bool FlashInferAttnParams::checkPrefill(rtp_llm::DeviceBase*             device,
                                        const rtp_llm::AttentionConfigs& attn_configs,
                                        const BufferPtr&                 prefix_lengths_host,
                                        const BufferPtr&                 input_lengths_host,
                                        DataType                         dtype,
                                        bool                             skip_no_prefix) {
    RTP_LLM_LOG_DEBUG("%s", attn_configs.DebugAttentionConfigStr());
    return check(device, attn_configs, dtype, true);
}

bool FlashInferAttnParams::checkDecode(rtp_llm::DeviceBase*             device,
                                       const rtp_llm::AttentionConfigs& attn_configs,
                                       DataType                         dtype) {
    return check(device, attn_configs, dtype, false);
}

ParamsPtr FlashInferAttnParams::prepare(rtp_llm::DeviceBase*             device,
                                        const rtp_llm::AttentionConfigs& attn_configs,
                                        const BufferPtr&                 prefix_lengths_host,
                                        const BufferPtr&                 sequence_lengths_host,
                                        const BufferPtr&                 input_lengths_host,
                                        const BufferPtr&                 kv_cache_block_id_host,
                                        const BufferPtr&                 kv_cache_block_id_device,
                                        DataType                         dtype,
                                        bool                             skip_no_prefix) {

    const int batch_size = input_lengths_host->shape()[0];
    // should not happend
    if (batch_size == 0) {
        return nullptr;
    }
    bool is_prefill = prefix_lengths_host != nullptr && prefix_lengths_host->size();
    // to underlay buffer dtype
    if (dtype == DataType::TYPE_QFP8_E4M3) {
        dtype = DataType::TYPE_FP8_E4M3;
    }
    // tmp double check when models py
    if (!check(device, attn_configs, dtype, is_prefill)) {
        return nullptr;
    }
    auto       cuda_device  = dynamic_cast<CudaDevice*>(device);
    MlaOpsType mla_ops_type = device->mla_ops_type;
    int        q_length     = -1;
    if (mla_ops_type == MlaOpsType::FLASH_MLA) {
        if (is_prefill) {
            // check if is MTP decode
            if (!sameQLength(input_lengths_host, batch_size, q_length) || q_length == -1 || q_length > 32) {
                mla_ops_type = MlaOpsType::FLASH_INFER;
            }
        } else {
            q_length = 1;  // deocde q_length always 1
        }
    }

    const int local_head_num    = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head     = attn_configs.size_per_head;
    const int group_size        = local_head_num / local_head_num_kv;
    const int tokens_per_block  = attn_configs.tokens_per_block;

    int input_token_num = 0;
    if (is_prefill) {
        input_token_num =
            std::accumulate(input_lengths_host->data<int>(), input_lengths_host->data<int>() + batch_size, 0);
    } else {
        input_token_num = input_lengths_host->shape()[0];
    }

    auto      params = FlashInferAttnParams::create(cuda_device,
                                               max(MIN_CACHE_BATCH_SIZE, batch_size),
                                               max(MIN_CACHE_INPUT_TOKEN_NUM, input_token_num),
                                               MIN_CACHE_PAGE_NUM);
    ParamsPtr ret(params, recycle);

    if (kv_cache_block_id_device) {
        params->kv_cache_block_id_d = Buffer2torchTensor(kv_cache_block_id_device, false);
    }
    params->mla_ops_type = mla_ops_type;
    params->dtype        = dtype;
    params->fillFlashInfer(prefix_lengths_host,
                           sequence_lengths_host,
                           input_lengths_host,
                           kv_cache_block_id_host,
                           batch_size,
                           tokens_per_block);
    params->refreshFlashInferBuf(cuda_device, batch_size, input_token_num);

    if (is_prefill || (group_size > 2 && skip_no_prefix)) {
        params->decode_plan = false;
    } else {
        params->decode_plan = true;
    }

    // Todo(tuowu): flashinfer: do not use partition-kv kernel for short sequence, when not using CUDAGraph .
    // check how short as `short sequence`.
    // bool enable_cuda_graph = device->initParams().hw_kernel_config.enable_cuda_graph;
    params->genPlan(batch_size,
                    q_length,
                    local_head_num,
                    local_head_num_kv,
                    size_per_head,
                    tokens_per_block,
                    attn_configs.kv_lora_rank,
                    attn_configs.use_mla,
                    reinterpret_cast<int64_t>(cuda_device->getStream()),
                    false);  // cuda_stream

    return ret;
}

void FlashInferAttnParams::run(const AttentionModuleParams& params,
                               const BufferPtr&             input_q,
                               const BufferPtr&             f16_out,
                               int64_t                      stream) {
    const int size_per_head = params.configs.size_per_head;
    auto      q             = Buffer2torchTensor(input_q, false);
    auto      k_cache       = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false).select(1, 0);
    auto      v_cache       = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false).select(1, 1);

    auto       softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale;
    at::Tensor out;
    if (params.output.type() == DataType::TYPE_FP8_E4M3) {
        out = Buffer2torchTensor(f16_out, false);
    } else {
        out = Buffer2torchTensor(params.output, false);
    }
    if (decode_plan) {
        RTP_LLM_LOG_DEBUG("decode flashinfer");
        BatchDecodeWithPagedKVCacheRun(float_workspace_d,         // float_workspace_buffer
                                       int_workspace_d,           // int_workspace_buffer
                                       plan,                      // plan_info_vec
                                       q,                         // q
                                       k_cache,                   // paged_k_cache
                                       v_cache,                   // paged_v_cache
                                       page_indptr_d,             // paged_kv_indptr
                                       page_indice_d,             // paged_kv_indices
                                       paged_kv_last_page_len_d,  // paged_kv_last_page_len
                                       out,
                                       std::nullopt,  // maybe_lse
                                       1,             // kv_layout_code
                                       -1,            // window_left
                                       std::nullopt,  // maybe_alibi_slopes
                                       0,             // logits_soft_cap
                                       softmax_scale,
                                       0,
                                       0,
                                       stream);
    } else {
        RTP_LLM_LOG_DEBUG("prefill flashinfer");
        // reference to flashinfer doc:
        // https://docs.flashinfer.ai/api/attention.html#flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run
        BatchPrefillWithPagedKVCacheRun(float_workspace_d,         // float_workspace_buffer
                                        int_workspace_d,           // int_workspace_buffer
                                        plan,                      // plan_info_vec
                                        q,                         // q
                                        k_cache,                   // paged_k_cache
                                        v_cache,                   // paged_v_cache
                                        qo_indptr_d,               // qo_indptr
                                        page_indptr_d,             // paged_kv_indptr
                                        page_indice_d,             // paged_kv_indices
                                        paged_kv_last_page_len_d,  // paged_kv_last_page_len
                                        out,
                                        std::nullopt,  // maybe_lse
                                        1,             // mask_mode_code,
                                        1,             // layout
                                        -1,            // window_left
                                        std::nullopt,  // maybe_custom_mask
                                        std::nullopt,  // maybe_mask_indptr
                                        std::nullopt,  // maybe_alibi_slopes
                                        0,             // logits_soft_cap
                                        softmax_scale,
                                        params.configs.rope_config.scale,
                                        params.configs.rope_config.base,
                                        stream);
    }
    if (params.configs.kv_cache_dtype == KvCacheDataType::FP8) {
        const auto& scale = params.weights.static_scale_reciprocal_weight;
        RTP_LLM_CHECK_WITH_INFO(scale != nullptr, "static_scale_reciprocal_weight is not set");
        auto scale_t = Buffer2torchTensor(scale->kernel, false);
        auto fp8_out = Buffer2torchTensor(params.output, false);
        fp8_out.copy_((scale_t * out).to(torch::kFloat8_e4m3fn));
    }
}

}  // namespace rtp_llm
