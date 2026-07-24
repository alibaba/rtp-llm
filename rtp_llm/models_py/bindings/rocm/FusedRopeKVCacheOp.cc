#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_rope_kvcache_kernel.h"
#include "rtp_llm/models_py/bindings/core/Dispatch.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include <stdexcept>
#include <string>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/model_utils/RopeCache.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {

static at::ScalarType get_fp8_dtype() {
    hipDeviceProp_t prop;
    int             device_id = 0;
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&prop, device_id);
    std::string arch(prop.gcnArchName);
    if (arch.find("gfx950") != std::string::npos) {
        return torch::kFloat8_e4m3fn;
    }
    return torch::kFloat8_e4m3fnuz;  // gfx942 and default
}

static void
validateMropePositionIds(const RopeConfig&, const torch::Tensor&, int64_t, const char*, bool require_device = true);

static void copyTensorExactInPlace(torch::Tensor& dst, const torch::Tensor& src, const char* name) {
    TORCH_CHECK(src.defined(), "prepare_in_place expects defined tensor: ", name);
    torch::Tensor src_flat = src.contiguous().reshape({-1});
    if (!dst.defined()) {
        dst = src_flat.clone();
        return;
    }
    torch::Tensor dst_flat = dst.reshape({-1});
    TORCH_CHECK(dst_flat.numel() == src_flat.numel(),
                "prepare_in_place tensor size mismatch for ",
                name,
                ": capture=",
                dst_flat.numel(),
                ", replay=",
                src_flat.numel());

    dst_flat.copy_(src_flat, /*non_blocking=*/true);
}

void updateKvCacheOffset(CKAttn& params, const torch::Tensor& kv_cache_block_id_device) {
    if (!params.kv_cache_offset.defined() || !kv_cache_block_id_device.defined()
        || kv_cache_block_id_device.numel() == 0) {
        return;
    }
    TORCH_CHECK(kv_cache_block_id_device.scalar_type() == at::kInt,
                "kv_cache_block_id_device must be int32, got ",
                kv_cache_block_id_device.scalar_type());
    const int   batch_size        = kv_cache_block_id_device.size(0);
    const int   max_blocks_per_bs = kv_cache_block_id_device.size(1);
    hipStream_t stream            = GET_CURRENT_STREAM();
    invokeConvertOffsetToBlockArrayData(params.kv_cache_offset.data_ptr<int>(),
                                        kv_cache_block_id_device.data_ptr<int>(),
                                        batch_size,
                                        max_blocks_per_bs,
                                        stream);
}

void prepareInPlace(CKAttn& params, const torch_ext::PyAttentionInputs& attn_inputs) {
    const bool has_prefix = attn_inputs.prefix_lengths.defined() && attn_inputs.prefix_lengths.numel() > 0;

    if (has_prefix && params.prefix_lengths.defined() && params.prefix_lengths.numel() > 0
        && params.prefix_lengths.data_ptr() != attn_inputs.prefix_lengths.data_ptr()) {
        copyTensorExactInPlace(params.prefix_lengths, attn_inputs.prefix_lengths, "prefix_lengths");
    }

    params.max_seq_len = attn_inputs.input_lengths.max().item<int32_t>();
    int max_prefix_len = 0;
    if (has_prefix) {
        max_prefix_len = attn_inputs.prefix_lengths.max().item<int32_t>();
    }
    params.prefill_runtime_max_seq_len         = params.max_seq_len;
    params.prefill_runtime_max_prefix_len      = max_prefix_len;
    params.prefill_runtime_seq_len_with_prefix = params.max_seq_len + max_prefix_len;

    const bool captured_position_ids = params.position_ids.defined() && params.position_ids.numel() > 0;
    const bool replay_position_ids =
        attn_inputs.combo_position_ids.defined() && attn_inputs.combo_position_ids.numel() > 0;
    if (captured_position_ids || replay_position_ids) {
        TORCH_CHECK(captured_position_ids && replay_position_ids,
                    "prepare_in_place requires combo_position_ids in both capture and replay inputs");
        torch::Tensor replay_ids = attn_inputs.combo_position_ids.contiguous();
        validateMropePositionIds(params.rope_config, replay_ids, -1, "prepare_in_place", false);
        if (params.position_ids.data_ptr() != replay_ids.data_ptr()) {
            // CUDA graph inputs allocate combo_position_ids as pinned host
            // memory. Copy directly into the persistent device capture buffer
            // on the current stream; avoid a synchronous per-replay .to().
            copyTensorExactInPlace(params.position_ids, replay_ids, "combo_position_ids");
        }
        validateMropePositionIds(params.rope_config, params.position_ids, -1, "prepare_in_place");
    }

    updateKvCacheOffset(params, attn_inputs.kv_cache_kernel_block_id_device);
}

static void validateMropePositionIds(const RopeConfig&    rope_config,
                                     const torch::Tensor& position_ids,
                                     int64_t              expected_tokens,
                                     const char*          where,
                                     bool                 require_device) {
    if (rope_config.style != RopeStyle::Mrope) {
        return;
    }

    TORCH_CHECK(rope_config.index_factor == 3,
                where,
                ": RopeStyle::Mrope requires index_factor == 3, got ",
                rope_config.index_factor);
    const int mrope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
    TORCH_CHECK(rope_config.mrope_dim1 > 0 && rope_config.mrope_dim2 > 0 && rope_config.mrope_dim3 > 0
                    && mrope_dim * 2 == rope_config.dim,
                where,
                ": invalid Mrope sections [",
                rope_config.mrope_dim1,
                ", ",
                rope_config.mrope_dim2,
                ", ",
                rope_config.mrope_dim3,
                "] for rope dim ",
                rope_config.dim);
    TORCH_CHECK(position_ids.defined() && position_ids.numel() > 0,
                where,
                ": RopeStyle::Mrope requires non-empty combo_position_ids");
    TORCH_CHECK(position_ids.scalar_type() == at::kInt,
                where,
                ": combo_position_ids must be int32, got ",
                position_ids.scalar_type());
    TORCH_CHECK(!require_device || position_ids.is_cuda(), where, ": combo_position_ids must be on the ROCm device");
    TORCH_CHECK(position_ids.is_contiguous(), where, ": combo_position_ids must be contiguous");
    TORCH_CHECK(position_ids.numel() % rope_config.index_factor == 0,
                where,
                ": combo_position_ids numel (",
                position_ids.numel(),
                ") must be divisible by index_factor (",
                rope_config.index_factor,
                ")");
    if (expected_tokens >= 0) {
        TORCH_CHECK(position_ids.numel() == expected_tokens * rope_config.index_factor,
                    where,
                    ": combo_position_ids numel mismatch: expected ",
                    expected_tokens * rope_config.index_factor,
                    " for ",
                    expected_tokens,
                    " tokens and index_factor ",
                    rope_config.index_factor,
                    ", got ",
                    position_ids.numel());
    }
}

FusedRopeKVCachePrefillOpBase::FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs) {}

FusedRopeKVCachePrefillOpAsm::FusedRopeKVCachePrefillOpAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCachePrefillOpBase(attn_configs) {}

FusedRopeKVCachePrefillOpNonAsm::FusedRopeKVCachePrefillOpNonAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCachePrefillOpBase(attn_configs) {}

CKAttnPtr FusedRopeKVCachePrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.input_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id.defined() && attn_inputs.kv_cache_kernel_block_id.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    bool has_prefix = attn_inputs.prefix_lengths.defined() && attn_inputs.prefix_lengths.numel() > 0;

    const bool use_fmha_fp8 = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    CKAttnPtr  attn_params;
    auto       params =
        PrepareCKAttn(attn_configs_, kv_cache_kernel_block_id_device, attn_inputs.input_lengths.size(0), use_fmha_fp8);
    if (params) {
        attn_params = CKAttnPtr(params, (CKAttn*)params.get());
    } else {
        attn_params = std::make_shared<CKAttn>();
    }
    attn_params->rope_config    = attn_configs_.rope_config;
    attn_params->attn_type      = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens     = attn_inputs.cu_seqlens_device;
    attn_params->cu_kv_seqlens  = attn_inputs.cu_kv_seqlens_device;
    attn_params->input_lengths  = attn_inputs.input_lengths;
    attn_params->max_seq_len    = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->padding_offset = attn_inputs.padding_offset;

    // 处理 prefix_lengths：确保在 CUDA 上且连续
    if (has_prefix) {
        torch::Tensor prefix_lengths = attn_inputs.prefix_lengths;
        if (!prefix_lengths.is_cuda()) {
            prefix_lengths = prefix_lengths.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
        }
        attn_params->prefix_lengths = prefix_lengths.contiguous();
    } else {
        attn_params->prefix_lengths = attn_inputs.prefix_lengths;
    }
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->position_ids              = attn_inputs.combo_position_ids;

    // MRoPE position IDs originate on the host in the normal engine. Keep a
    // contiguous device tensor because the fused kernel indexes the flattened
    // token-major [token, axis] layout directly.
    if (attn_params->position_ids.defined()) {
        if (!attn_params->position_ids.is_cuda()) {
            attn_params->position_ids =
                attn_params->position_ids.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
        }
        attn_params->position_ids = attn_params->position_ids.contiguous();
    }
    validateMropePositionIds(
        attn_configs_.rope_config, attn_params->position_ids, -1, "FusedRopeKVCachePrefillOp::prepare");

    int max_prefix_length = 0;
    if (has_prefix && attn_params->prefix_lengths.defined() && attn_params->prefix_lengths.numel() > 0) {
        max_prefix_length = attn_params->prefix_lengths.max().item<int32_t>();
    }
    attn_params->prefill_runtime_max_seq_len         = attn_params->max_seq_len;
    attn_params->prefill_runtime_max_prefix_len      = max_prefix_length;
    attn_params->prefill_runtime_seq_len_with_prefix = attn_params->max_seq_len + max_prefix_length;
    return attn_params;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FusedRopeKVCachePrefillOpBase::forward(
    const torch::Tensor& qkv, std::optional<torch_ext::LayerKVCache> kv_cache, const CKAttnPtr& params) {
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;
    const int seq_len =
        params->prefill_runtime_max_seq_len >= 0 ? params->prefill_runtime_max_seq_len : params->max_seq_len;
    const int max_prefix_length =
        params->prefill_runtime_max_prefix_len >= 0 ? params->prefill_runtime_max_prefix_len : 0;
    const int seq_len_with_prefix = seq_len + max_prefix_length;
    validateMropePositionIds(
        attn_configs_.rope_config, params->position_ids, token_num, "FusedRopeKVCachePrefillOp::forward");

    const int  q_output_token_num = (use_paged_fmha && pad_query) ? batch_size * seq_len : token_num;
    const bool paged_fp8          = use_paged_fmha && attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    const auto q_opts             = torch::TensorOptions(qkv.dtype()).device(qkv.device());

    // pad_query=false: q_output is packed [token_num, heads, dim] and the kernel writes
    // every cell — skip the zero-fill. pad_query=true: padded slots between sequences
    // are not written by the kernel, so they must be zero-initialized for downstream
    // FMHA correctness.
    torch::Tensor q_output = (use_paged_fmha && pad_query) ?
                                 torch::zeros({q_output_token_num, local_head_num, size_per_head}, q_opts) :
                                 torch::empty({q_output_token_num, local_head_num, size_per_head}, q_opts);
    torch::Tensor q_fp8_buf;
    if (paged_fp8) {
        q_fp8_buf = torch::empty({q_output_token_num, local_head_num, size_per_head},
                                 torch::TensorOptions(get_fp8_dtype()).device(qkv.device()));
    }

    PrefixPromptBatchWeightsParam prefix_prompt_param{};
    bool                          use_fmha_fp8 = false;
    if (kv_cache.has_value()) {
        // 验证KV cache指针有效性
        if (!kv_cache.value().kv_cache_base.defined() || kv_cache.value().kv_cache_base.numel() == 0) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: kv_cache_base is not defined or empty");
        }

        auto  kv_block_array = params->kv_block_array;
        void* k_cache_ptr    = kv_cache.value().kv_cache_base.data_ptr();

        kv_block_array.mPrimaryPoolPtr = k_cache_ptr;
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
            void* scale_ptr = kv_cache.value().kv_scale_base.data_ptr();
            if (scale_ptr != nullptr) {
                kv_block_array.scale = scale_ptr;
            }
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        use_fmha_fp8                       = kv_block_array.cache_type == KvCacheDataType::FP8;
    }

    // 设置 prefix_lengths 参数
    if (max_prefix_length > 0) {
        if (!params->prefix_lengths.defined() || params->prefix_lengths.numel() < batch_size) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: prefix_lengths is not ready for runtime replay");
        }
        int* prefix_lengths_ptr = params->prefix_lengths.data_ptr<int>();

        prefix_prompt_param.d_prefix_prompt_lengths  = prefix_lengths_ptr;
        prefix_prompt_param.max_prefix_prompt_length = max_prefix_length;
        prefix_prompt_param.count_length             = 1;
    }

    if (qkv.dtype().toScalarType() == torch::kFloat16) {
        // TODO: FP8 FMHA currently does not support FP16 output.
        //       Please run with BF16 activation instead (set environment variable ACT_TYPE=bf16)
        use_fmha_fp8 = false;
    }
    // FP8 path: keep original behavior (store QKV linearly for flash_attn_varlen_fp8).
    // Non-FP8 with paged cache: K/V go directly into the cache via store_cache, so
    //   store_kv=false (writing k_output/v_output would be wasted HBM bandwidth).
    // Non-FP8 without paged cache (embedding models): store_kv=true so K/V are
    //   returned as padded buffers for downstream varlen attention; RoPE must still
    //   run for positional encoding.
    bool store_qkv   = use_fmha_fp8 ? !use_paged_fmha : false;
    bool store_q     = true;
    bool store_cache = kv_cache.has_value();
    bool store_kv    = use_fmha_fp8 ? !use_paged_fmha : !store_cache;

    // Allocate K/V output buffers only when the kernel actually writes them,
    // avoiding unnecessary GPU memory allocation and zero-fill.
    torch::Tensor k_output;
    torch::Tensor v_output;
    if (store_kv) {
        k_output = torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head}, q_opts);
        v_output = torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head}, q_opts);
    }
    void* k_output_ptr = store_kv ? k_output.data_ptr() : nullptr;
    void* v_output_ptr = store_kv ? v_output.data_ptr() : nullptr;

    // int8
    float* scale_out_ptr = nullptr;
    int    int8_mode     = 0;
    // Always use aiter_pa for ROCm
    hipStream_t stream_ = GET_CURRENT_STREAM();
    // 添加 FP8 缓冲区支持
    torch::Tensor qkv_buf_fp8;
    if (use_fmha_fp8 && !paged_fp8) {
        qkv_buf_fp8 = torch::empty(qkv.sizes(), torch::TensorOptions(get_fp8_dtype()).device(qkv.device()));
    }

    int *padding_offset = nullptr, *position_ids = nullptr;
    if (params->padding_offset.defined() && params->padding_offset.numel() > 0) {
        padding_offset = params->padding_offset.data_ptr<int>();
    }
    if (params->position_ids.defined()) {
        position_ids = params->position_ids.data_ptr<int>();
    }

    if (use_asm()) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTransposePrefill,
            q_output.data_ptr(),
            k_output_ptr,
            v_output_ptr,
            &prefix_prompt_param,
            qkv.data_ptr(),
            paged_fp8 ? q_fp8_buf.data_ptr() :
                        (use_fmha_fp8 && qkv_buf_fp8.defined() ? qkv_buf_fp8.data_ptr() : nullptr),
            position_ids,
            nullptr,  // qkv_bias
            padding_offset,
            params->cu_seqlens.data_ptr<int>(),
            batch_size,
            seq_len,
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            scale_out_ptr,
            int8_mode,
            use_fmha_fp8 ? use_paged_fmha : true,  // FP8: original flag; non-FP8: always paged
            store_qkv,
            store_q,
            store_kv,
            store_cache,
            nullptr,
            pad_query,
            stream_);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTransposePrefillV1,
            q_output.data_ptr(),
            k_output_ptr,
            v_output_ptr,
            &prefix_prompt_param,
            qkv.data_ptr(),
            paged_fp8 ? q_fp8_buf.data_ptr() :
                        (use_fmha_fp8 && qkv_buf_fp8.defined() ? qkv_buf_fp8.data_ptr() : nullptr),
            position_ids,
            nullptr,
            padding_offset,
            params->cu_seqlens.data_ptr<int>(),
            batch_size,
            seq_len,
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,
            0,
            use_fmha_fp8 ? use_paged_fmha : true,  // FP8: original flag; non-FP8: always paged
            store_qkv,
            store_q,
            store_kv,
            store_cache,
            nullptr,
            stream_);
    }
    // FP8 path: paged returns Q-only fp8 buf; non-paged returns full qkv fp8 buf
    if (use_fmha_fp8) {
        return std::make_tuple(paged_fp8 ? q_fp8_buf : qkv_buf_fp8, torch::Tensor(), torch::Tensor());
    }
    // Non-FP8 with paged cache: return bf16 Q (K/V are already written into the cache).
    // Non-FP8 without paged cache: also return padded K/V for flash_attn_varlen_func.
    return std::make_tuple(q_output, k_output, v_output);
}

FusedRopeKVCacheDecodeOpBase::FusedRopeKVCacheDecodeOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs) {}

FusedRopeKVCacheDecodeOpAsm::FusedRopeKVCacheDecodeOpAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCacheDecodeOpBase(attn_configs) {}

FusedRopeKVCacheDecodeOpNonAsm::FusedRopeKVCacheDecodeOpNonAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCacheDecodeOpBase(attn_configs) {}

CKAttnPtr FusedRopeKVCacheDecodeOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.sequence_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id.defined() && attn_inputs.kv_cache_kernel_block_id.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    CKAttnPtr attn_params;
    bool      use_fmha_fp8 = false;
    use_fmha_fp8           = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;

    auto params = PrepareCKAttn(
        attn_configs_, kv_cache_kernel_block_id_device, attn_inputs.sequence_lengths.size(0), use_fmha_fp8);
    if (!params) {
        throw std::runtime_error("FusedRopeKVCacheDecodeOp::prepare: PrepareCKAttn failed. "
                                 "kv_cache_kernel_block_id_size="
                                 + std::to_string(attn_inputs.kv_cache_kernel_block_id.size(0)));
    }

    attn_params                            = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->rope_config               = attn_configs_.rope_config;
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens_device;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens_device;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->input_lengths             = attn_inputs.input_lengths;
    attn_params->prefix_lengths            = attn_inputs.prefix_lengths;
    attn_params->padding_offset            = attn_inputs.padding_offset;
    attn_params->position_ids              = attn_inputs.combo_position_ids;
    if (attn_params->position_ids.defined()) {
        if (!attn_params->position_ids.is_cuda()) {
            attn_params->position_ids =
                attn_params->position_ids.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
        }
        attn_params->position_ids = attn_params->position_ids.contiguous();
    }
    validateMropePositionIds(
        attn_configs_.rope_config, attn_params->position_ids, -1, "FusedRopeKVCacheDecodeOp::prepare");

    if (attn_inputs.kv_cache_kernel_block_id_device.defined()
        && attn_inputs.kv_cache_kernel_block_id_device.numel() > 0) {
        attn_params->kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOpBase::forward(const torch::Tensor&                   qkv,
                                                    std::optional<torch_ext::LayerKVCache> kv_cache,
                                                    const CKAttnPtr&                       params) {
    TORCH_CHECK(kv_cache.has_value(), "FusedRopeKVCacheDecodeOp::forward: decode should have kv cache");

    auto kv_block_array            = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
    if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel()) {
        kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
    }

    const int     local_head_num    = attn_configs_.head_num;
    const int     local_head_num_kv = attn_configs_.kv_head_num;
    const int     size_per_head     = attn_configs_.size_per_head;
    const int     token_num         = qkv.size(0);
    const int     batch_size        = params->sequence_lengths.size(0);
    torch::Tensor q_output          = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    validateMropePositionIds(
        attn_configs_.rope_config, params->position_ids, token_num, "FusedRopeKVCacheDecodeOp::forward");

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    prefix_prompt_param.kv_block_array = kv_block_array;

    // 设置 prefix_lengths 参数
    int max_prefix_length = 0;
    if (params->prefix_lengths.defined() && params->prefix_lengths.size(0) > 0) {
        max_prefix_length = params->prefix_lengths.max().item<int>();

        int* prefix_lengths_ptr = params->prefix_lengths.data_ptr<int>();
        if (prefix_lengths_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCacheDecodeOp: prefix_lengths data pointer is null");
        }

        prefix_prompt_param.d_prefix_prompt_lengths  = prefix_lengths_ptr;
        prefix_prompt_param.max_prefix_prompt_length = max_prefix_length;
        prefix_prompt_param.count_length             = 1;
    }

    size_t seq_len     = 1;
    bool   store_qkv   = false;
    bool   store_q     = true;
    bool   store_kv    = false;
    bool   store_cache = kv_cache.has_value();

    // Always use aiter_pa for ROCm
    hipStream_t stream_ = GET_CURRENT_STREAM();

    int* position_ids_ptr = nullptr;
    if (params->position_ids.defined()) {
        position_ids_ptr = params->position_ids.data_ptr<int>();
    } else {
        position_ids_ptr = params->sequence_lengths.data_ptr<int>();
    }

    auto    rope_cache = getRopeCacheOnce(attn_configs_.rope_config, attn_configs_.max_seq_len, false);
    float2* rope_cache_ptr =
        rope_cache.used && rope_cache.data.defined() ? static_cast<float2*>(rope_cache.data.data_ptr()) : nullptr;

    if (use_asm()) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposeDecode,
                                         q_output.data_ptr(),
                                         nullptr,
                                         nullptr,
                                         &prefix_prompt_param,
                                         params->input_lengths.data_ptr<int>(),
                                         qkv.data_ptr(),
                                         nullptr,
                                         position_ids_ptr,
                                         /*qkv_bias*/ nullptr,
                                         params->padding_offset.data_ptr<int>(),
                                         params->cu_seqlens.data_ptr<int>(),
                                         params->sequence_lengths.data_ptr<int>(),
                                         batch_size,
                                         seq_len,
                                         token_num,
                                         local_head_num,
                                         local_head_num_kv,
                                         size_per_head,
                                         attn_configs_.rope_config,
                                         attn_configs_.use_logn_attn,
                                         nullptr,
                                         0,
                                         false,
                                         store_qkv,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         rope_cache_ptr,
                                         stream_);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposeDecodeV1,
                                         q_output.data_ptr(),
                                         nullptr,
                                         nullptr,
                                         &prefix_prompt_param,
                                         params->input_lengths.data_ptr<int>(),
                                         qkv.data_ptr(),
                                         nullptr,
                                         position_ids_ptr,
                                         /*qkv_bias*/ nullptr,
                                         params->padding_offset.data_ptr<int>(),
                                         params->cu_seqlens.data_ptr<int>(),
                                         params->sequence_lengths.data_ptr<int>(),
                                         batch_size,
                                         seq_len,
                                         token_num,
                                         local_head_num,
                                         local_head_num_kv,
                                         size_per_head,
                                         attn_configs_.rope_config,
                                         attn_configs_.use_logn_attn,
                                         nullptr,
                                         0,
                                         false,
                                         store_qkv,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         rope_cache_ptr,
                                         stream_);
    }

    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<CKAttn, std::shared_ptr<CKAttn>>(m, "CKAttn")
        .def(pybind11::init<>())
        .def("update_kv_cache_offset", &updateKvCacheOffset, py::arg("kv_cache_block_id_device"))
        .def("prepare_in_place", &prepareInPlace, py::arg("attn_inputs"));

    // Prefill ASM
    pybind11::class_<FusedRopeKVCachePrefillOpAsm>(m, "FusedRopeKVCachePrefillOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def_readwrite("use_paged_fmha", &FusedRopeKVCachePrefillOpAsm::use_paged_fmha)
        .def_readwrite("pad_query", &FusedRopeKVCachePrefillOpAsm::pad_query)
        .def("prepare", &FusedRopeKVCachePrefillOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward", &FusedRopeKVCachePrefillOpAsm::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));

    // Prefill Non-ASM
    pybind11::class_<FusedRopeKVCachePrefillOpNonAsm>(m, "FusedRopeKVCachePrefillOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def_readwrite("use_paged_fmha", &FusedRopeKVCachePrefillOpNonAsm::use_paged_fmha)
        .def_readwrite("pad_query", &FusedRopeKVCachePrefillOpNonAsm::pad_query)
        .def("prepare", &FusedRopeKVCachePrefillOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("kv_cache"),
             py::arg("params"));

    // Decode ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpAsm>(m, "FusedRopeKVCacheDecodeOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward", &FusedRopeKVCacheDecodeOpAsm::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));

    // Decode Non-ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpNonAsm>(m, "FusedRopeKVCacheDecodeOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("kv_cache"),
             py::arg("params"));
}
}  // namespace rtp_llm
