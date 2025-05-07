
#include <iostream>
#include <numeric>
#include "maga_transformer/cpp/devices/OpData.h"
#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "maga_transformer/cpp/devices/CommonDefines.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include "maga_transformer/cpp/utils/compiler_config.h"
#include "maga_transformer/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "maga_transformer/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "flashmla/flashmla.h"

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

using Slice = torch::indexing::Slice;
constexpr auto TNone = torch::indexing::None;

static std::deque<FlashInferAttnParams*> PARAMS_CACHE;
static int MIN_CACHE_BATCH_SIZE = 256;
static int MIN_CACHE_INPUT_TOKEN_NUM = 512;
static int MIN_CACHE_PAGE_NUM = 48 * 1024;

void FlashInferAttnParamsDel(void* p) {
    PARAMS_CACHE.push_back((FlashInferAttnParams *)p);
}

tuple<BufferPtr, vector<torch::Tensor>> FlashInferAttnParams::allocateManyBuffer(
        CudaDevice *device,
        const std::vector<std::vector<int64_t>> &shapes, AllocationType atype)
{
    vector<torch::Tensor> tensors;
    vector<size_t> sizes;
    size_t total_size = 0;
    for (const auto &shape : shapes) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= dim;
        }
        size = (size + 31) / 32 * 32;
        sizes.push_back(size);
        total_size += size;
    }
    auto buf = device->allocateBuffer({DataType::TYPE_INT32, {total_size}, atype}, {"flashinfer_buf"});
    auto buf_ptr = buf->data<int>();
    auto cuda_option = torch::dtype(torch::kInt).device(torch::DeviceType::CUDA).requires_grad(false);

    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        tensors.emplace_back(torch::from_blob(buf_ptr + offset, shapes[i], cuda_option));
        offset += sizes[i];
    }
    return {buf, tensors};
}

FlashInferAttnParams *FlashInferAttnParams::create(CudaDevice *device, int batch_size, int input_token_num, int page_num) {
    if (!PARAMS_CACHE.empty()) {
        auto params = PARAMS_CACHE.back();
        PARAMS_CACHE.pop_back();
        if (batch_size < params->batch_size &&
            input_token_num < params->input_token_num)
        {
            return params;
        }
        delete params;
    }

    auto params = make_unique<FlashInferAttnParams>();
    params->batch_size = batch_size;
    params->input_token_num = input_token_num;
    params->page_num = page_num;

    // batch_prefill_tmp_v may use 256M buffer
    params->float_workspace = device->allocateBuffer({DataType::TYPE_INT8, {(256 + 16) * 1024 * 1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 *1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::HOST}, {"int_host_workspace"});

    params->float_workspace_d = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_d = Buffer2torchTensor(params->int_workspace, false);
    params->int_workspace_h = Buffer2torchTensor(params->int_host_workspace, false);

#define ALLOC_BUFFER(suffix, type)                              \
    do {                                                        \
        auto alloc_ret = allocateManyBuffer(device, {           \
                {batch_size + 1}, /* page_indptr */             \
                {batch_size + 1}, /* qo_indptr */               \
                {input_token_num},     /* batch_indice */       \
                {input_token_num},     /* positions */          \
                {batch_size},     /* kv_len */                  \
                {batch_size},     /* paged_kv_last_page_len */  \
                {page_num}},      /* page_indice */             \
            type);                                              \
                                                                \
        params->buf_##suffix = get<0>(alloc_ret);               \
        auto &tensors = get<1>(alloc_ret);                      \
        params->page_indptr_##suffix = tensors[0];              \
        params->qo_indptr_##suffix = tensors[1];                \
        params->batch_indice_##suffix = tensors[2];             \
        params->positions_##suffix = tensors[3];                \
        params->kvlen_##suffix = tensors[4];                    \
        params->paged_kv_last_page_len_##suffix = tensors[5];   \
        params->page_indice_##suffix = tensors[6];              \
    } while (0)

    ALLOC_BUFFER(h, AllocationType::HOST);
    ALLOC_BUFFER(d, AllocationType::DEVICE);

    return params.release();
}

void FlashInferAttnParams::fillFlashInfer(const BufferPtr &prefix_lengths_host,
                                          const BufferPtr &sequence_lengths_host,
                                          const BufferPtr &input_lengths_host,
                                          const BufferPtr &kv_cache_block_id_host,
                                          const int batch_size,
                                          const int tokens_per_block)
{
    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    RTP_LLM_CHECK_WITH_INFO(batch_size <= this->batch_size, "batch_size exceed reserved %d > %d", batch_size, this->batch_size);

    auto qo_indptr = qo_indptr_h.data_ptr<int>();
    auto page_indptr = page_indptr_h.data_ptr<int>();
    auto batch_indice = batch_indice_h.data_ptr<int>();
    auto positions = positions_h.data_ptr<int>();
    auto paged_kv_last_page_len = paged_kv_last_page_len_h.data_ptr<int>();
    auto kvlen = kvlen_h.data_ptr<int>();
    auto page_indice = page_indice_h.data_ptr<int>();

    auto input_lengths = input_lengths_host->data<int>();
    auto sequence_lengths = sequence_lengths_host ? sequence_lengths_host->data<int>() : nullptr;
    auto prefix_lengths = prefix_lengths_host ? prefix_lengths_host->data<int>() : nullptr;
    auto kv_cache_block_id = kv_cache_block_id_host ? kv_cache_block_id_host->data<int>() : nullptr;

    int qo_offset = 0;
    int offset = 0;
    int total_page_idx = 0;
    qo_indptr[0] = 0;
    page_indptr[0] = 0;
    for (int i = 0; i < batch_size; i++) {
        int seq_len = 0;
        if (prefix_lengths) {
            int input_length = input_lengths[i];
            int prefix_length = prefix_lengths[i];
            RTP_LLM_CHECK_WITH_INFO(offset + input_length <= this->input_token_num, "token_num exceed reserved %d > %d",
                               offset + input_length, this->input_token_num);
            for (int j = 0; j < input_length; j++) {
                batch_indice[offset] = i;
                positions[offset] = j + prefix_length;
                offset += 1;
            }
            qo_offset += input_length;
            seq_len = input_length + prefix_length;
        } else {
            batch_indice[i] = i;
            positions[i] = sequence_lengths[i];
            qo_offset += 1;
            seq_len = sequence_lengths[i] + 1;
        }

        paged_kv_last_page_len[i] = (seq_len - 1) %  tokens_per_block + 1;
        kvlen[i] = seq_len;

        int page_num = (seq_len + tokens_per_block - 1) / tokens_per_block;
        RTP_LLM_CHECK_WITH_INFO(total_page_idx + page_num <= this->page_num, "page_num exceed reserved %d > %d",
                           total_page_idx + page_num, this->page_num);
        if (kv_cache_block_id) {
            for (int j = 0; j < page_num; j++) {
                auto page_idx = kv_cache_block_id[i * max_batch_blocks + j];
                page_indice[total_page_idx++] = page_idx;
            }
        }
        page_indptr[i + 1] = total_page_idx;
        qo_indptr[i + 1] = qo_offset;
    }
}

void FlashInferAttnParams::refreshFlashInferBuf(CudaDevice *device, int batch_size, int input_token_num) {
    auto stream = device->getStream();
    cudaMemcpyAsync(buf_d->data(), buf_h->data(), buf_h->sizeBytes(), cudaMemcpyHostToDevice, stream);

    vector<int64_t> shape = {batch_size + 1};
#define REFRESH_SHAPE(t)                                                \
    do {                                                                \
        t##_d.unsafeGetTensorImpl()->set_sizes_contiguous(shape);        \
        t##_h.unsafeGetTensorImpl()->set_sizes_contiguous(shape);       \
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

bool FlashInferAttnParams::sameQLength(const BufferPtr &input_lengths_host, int batch_size, int &q_length) {
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


void FlashInferAttnParams::genPlan(int batch_size,
                                   int q_length,
                                   int local_head_num,
                                   int local_head_num_kv,
                                   int size_per_head,
                                   int tokens_per_block,
                                   int kv_lora_rank,
                                   bool use_mla,
                                   int64_t stream)
{
    // std::cout << "use_mla: " << use_mla << std::endl
    //           << "mla_type: " << int(mla_ops_type) << std::endl
    //           << "page_indptr: " << page_indptr_d
    //           << "qo_indptr: " << qo_indptr_d
    //           << "batch_indice: " << batch_indice_d
    //           << "positions: " << positions_d
    //           << "kvlen: " << kvlen_d
    //           << "paged_kv_last_page_len: " << paged_kv_last_page_len_d
    //           << "page_indice: " << page_indice_d.index({torch::indexing::Slice(0, 32)})
    //           << "kv_cache_block_id: " << kv_cache_block_id_d << std::endl;

    if (use_mla) {
        if (mla_ops_type == MlaOpsType::FLASH_INFER) {
            plan = BatchMLAPagedAttentionPlan(
                    float_workspace_d,
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
            RTP_LLM_LOG_TRACE("batch_size = %zu", batch_size);
            RTP_LLM_LOG_TRACE("local_head_num = %zu", local_head_num);
            flash_mla_plan = get_mla_metadata(kvlen_d, local_head_num * q_length, 1);
        } else {
            RTP_LLM_FAIL("unexpected mla ops type: %d", int(mla_ops_type));
        }
    } else {
        if (decode) {
            plan = BatchDecodeWithPagedKVCachePlan(
                    float_workspace_d, // float_workspace_buffer
                    int_workspace_d, // int_workspace_buffer
                    int_workspace_h, // page_locked_int_workspace_buffer
                    page_indptr_h, // indptr
                    batch_size, // batch_size
                    local_head_num, // num_qo_heads
                    local_head_num_kv, // num_kv_heads
                    tokens_per_block, // page_size
                    false, // enable_cuda_graph,
                    -1, // window_left
                    -1, // logits_soft_cap
                    size_per_head, // head_dim_qk
                    size_per_head, // head_dim_vo
                    torch::empty(0, dataTypeToTorchType(dtype)), // empty_q_data
                    torch::empty(0, dataTypeToTorchType(dtype)), // empty_kv_data
                    stream);
        } else {
            plan = BatchPrefillWithKVCachePlan(
                    float_workspace_d, // float_workspace_buffer
                    int_workspace_d, // int_workspace_buffer
                    int_workspace_h, // page_locked_int_workspace_buffer
                    qo_indptr_h, // qo_indptr
                    page_indptr_h, // kv_indptr
                    torch::empty(0, dataTypeToTorchType(DataType::TYPE_INT32)), // kv_len_arr, not in use yet
                    batch_size, // total_num_rows
                    batch_size, // batch_size
                    local_head_num, // num_qo_heads
                    local_head_num_kv, // num_kv_heads
                    tokens_per_block, // page_size
                    false, // enable_cuda_graph
                    size_per_head, // head_dim_qk
                    size_per_head, // head_dim_vo
                    true, // causal
                    stream);
        }
    }
}

FlashInferAttnParamsPtr FlashInferAttnParams::prepare(
        rtp_llm::DeviceBase *device,
        const rtp_llm::AttentionConfigs &attn_configs,
        const BufferPtr &prefix_lengths_host,
        const BufferPtr &sequence_lengths_host,
        const BufferPtr &input_lengths_host,
        const BufferPtr &kv_cache_block_id_host,
        const BufferPtr &kv_cache_block_id_device,
        rtp_llm::DataType dtype)
{
    if (rtp_llm::get_sm() < 80) {
        return nullptr;
    }

    const int batch_size = input_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto cuda_device = dynamic_cast<CudaDevice*>(device);
    if (!cuda_device) {
        return nullptr;
    }

    MlaOpsType mla_ops_type = device->mla_ops_type;
    int q_length = -1;
    if (mla_ops_type == MlaOpsType::FLASH_MLA &&
        (!sameQLength(input_lengths_host, batch_size, q_length) || q_length == -1 || q_length > 32)) {
        mla_ops_type = MlaOpsType::FLASH_INFER;
    }

    const char* disable_flash_infer_env = getenv("DISABLE_FLASH_INFER");
    const bool disable_flash_infer (disable_flash_infer_env && strcmp(disable_flash_infer_env, "1") == 0);
    if ((!attn_configs.use_mla || mla_ops_type == MlaOpsType::FLASH_INFER) && disable_flash_infer) {
        return nullptr;
    }

    const int local_head_num    = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head = attn_configs.size_per_head;
    const int group_size = local_head_num / local_head_num_kv;
    const int tokens_per_block = attn_configs.tokens_per_block;

    // to underlay buffer dtype
    if (dtype == DataType::TYPE_QFP8_E4M3) {
        dtype = DataType::TYPE_FP8_E4M3;
    }

    if (!attn_configs.use_mla) {
        if ((dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16 && dtype != DataType::TYPE_FP8_E4M3) ||
            (attn_configs.kv_cache_dtype != KvCacheDataType::BASE &&
             !(attn_configs.kv_cache_dtype == KvCacheDataType::FP8 && dtype == DataType::TYPE_FP8_E4M3)) ||
            (attn_configs.rope_config.style != RopeStyle::Base && attn_configs.rope_config.style != RopeStyle::No)  ||
            attn_configs.mask_type != causalMask ||
            attn_configs.q_scaling != 1.0f ||
            attn_configs.use_logn_attn ||
            (size_per_head != 64 && size_per_head != 128 && size_per_head != 192) ||
            (group_size > 10 && group_size != 16))
        {
            return nullptr;
        }
    }

    int input_token_num = 0;
    if (prefix_lengths_host) {
        input_token_num = std::accumulate(input_lengths_host->data<int>(),
                                          input_lengths_host->data<int>() + batch_size,
                                          0);
    } else {
        input_token_num = input_lengths_host->shape()[0];
    }

    auto params = FlashInferAttnParams::create(cuda_device,
                                               max(MIN_CACHE_BATCH_SIZE, batch_size),
                                               max(MIN_CACHE_INPUT_TOKEN_NUM, input_token_num),
                                               MIN_CACHE_PAGE_NUM);
    FlashInferAttnParamsPtr ret(params, FlashInferAttnParamsDel);

    if (kv_cache_block_id_device) {
        params->kv_cache_block_id_d = Buffer2torchTensor(kv_cache_block_id_device, false);
    }
    params->mla_ops_type = mla_ops_type;
    params->dtype = dtype;
    params->fillFlashInfer(prefix_lengths_host,
                           sequence_lengths_host,
                           input_lengths_host,
                           kv_cache_block_id_host,
                           batch_size,
                           tokens_per_block);
    params->refreshFlashInferBuf(cuda_device, batch_size, input_token_num);

    if (group_size > 5) {
        params->decode = false;
    } else {
        params->decode = true;
    }

    params->genPlan(batch_size,
                    q_length,
                    local_head_num,
                    local_head_num_kv,
                    size_per_head,
                    tokens_per_block,
                    attn_configs.kv_lora_rank,
                    attn_configs.use_mla,
                    reinterpret_cast<int64_t>(cuda_device->getStream())); // cuda_stream

    return ret;
}

void FlashInferAttnParams::run(
        const AttentionModuleParams& params,
        const BufferPtr &f16_out,
        std::function<void()> moe_insertion_callback,
        int64_t stream)
{
    const int local_head_num = params.configs.head_num;
    const int local_head_num_kv = params.configs.kv_head_num;
    const int size_per_head = params.configs.size_per_head;

    if (params.weights.qkv_weight->bias) {
        at::Tensor qkv_input = Buffer2torchTensor(params.input, false);
        qkv_input.add_(Buffer2torchTensor(params.weights.qkv_weight->bias, false));
    }

    const int bs = params.input.shape()[0];
    const vector<int64_t> strides = {(local_head_num + 2 * local_head_num_kv) * size_per_head, size_per_head, 1};
    const auto cuda_option = torch::dtype(dataTypeToTorchType(params.input.type())).device(torch::DeviceType::CUDA).requires_grad(false);

    auto q = torch::from_blob(params.input.data(),
                              {bs, local_head_num, size_per_head},
                              strides, cuda_option);
    auto append_k = torch::from_blob(params.input.dataWithOffset(local_head_num * size_per_head),
                                     {bs, local_head_num_kv, size_per_head},
                                     strides, cuda_option);
    apply_rope_pos_ids(q,
                       append_k,
                       q,
                       append_k,
                       positions_d,
                       params.configs.rope_config.dim,
                       false,
                       params.configs.rope_config.scale,
                       params.configs.rope_config.base,
                       stream);
    sync_check_cuda_error();

    auto append_v = torch::from_blob(params.input.dataWithOffset((local_head_num + local_head_num_kv) * size_per_head),
                                     {bs, local_head_num_kv, size_per_head},
                                     strides, cuda_option);

    auto k_cache = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto v_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);
    if (append_k.type() != k_cache.type()) {
        append_k = append_k.to(k_cache.type());
        append_v = append_v.to(k_cache.type());
    }
    append_paged_kv_cache(append_k,
                          append_v,
                          batch_indice_d,
                          positions_d,
                          k_cache,
                          v_cache,
                          page_indice_d,
                          page_indptr_d,
                          paged_kv_last_page_len_d,
                          1,
                          stream);

    moe_insertion_callback();

    sync_check_cuda_error();

    auto softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale;
    at::Tensor out;
    if (params.output.type() == DataType::TYPE_FP8_E4M3) {
        out = Buffer2torchTensor(f16_out, false);
    } else {
        out = Buffer2torchTensor(params.output, false);
    }
    if (decode) {
        BatchDecodeWithPagedKVCacheRun(
                float_workspace_d, // float_workspace_buffer
                int_workspace_d, // int_workspace_buffer
                plan, // plan_info_vec
                q, // q
                k_cache, // paged_k_cache
                v_cache, // paged_v_cache
                page_indptr_d, // paged_kv_indptr
                page_indice_d, // paged_kv_indices
                paged_kv_last_page_len_d, // paged_kv_last_page_len
                out,
                std::nullopt, // maybe_lse
                1, // kv_layout_code
                -1, // window_left
                std::nullopt, // maybe_alibi_slopes
                0, // logits_soft_cap
                softmax_scale,
                0,
                0,
                stream);
    } else {
        BatchPrefillWithPagedKVCacheRun(
                float_workspace_d, // float_workspace_buffer
                int_workspace_d,  // int_workspace_buffer
                plan, // plan_info_vec
                q, // q
                k_cache, // paged_k_cache
                v_cache, // paged_v_cache
                qo_indptr_d, // qo_indptr
                page_indptr_d, // paged_kv_indptr
                page_indice_d, // paged_kv_indices
                paged_kv_last_page_len_d, // paged_kv_last_page_len
                out,
                std::nullopt, // maybe_lse
                1, // mask_mode_code,
                1, // layout
                -1, // window_left
                std::nullopt, // maybe_custom_mask
                std::nullopt, // maybe_mask_indptr
                std::nullopt, // maybe_alibi_slopes
                0, // logits_soft_cap
                softmax_scale,
                params.configs.rope_config.scale,
                params.configs.rope_config.base,
                stream);
    }

    const auto &scale = params.weights.static_scale_reciprocal_weight;
    if (scale) {
        auto scale_t = Buffer2torchTensor(scale->kernel, false);
        auto fp8_out = Buffer2torchTensor(params.output, false);
        fp8_out.copy_((scale_t * out).to(torch::kFloat8_e4m3fn));
    }
}

}
