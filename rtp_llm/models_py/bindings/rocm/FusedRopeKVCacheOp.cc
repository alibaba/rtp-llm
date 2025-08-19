#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/cuda/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
FusedRopeKVCachePrefillOp::FusedRopeKVCachePrefillOp(const GptInitParameter& gpt_init_parameter):
    FMHARocmBase(gpt_init_parameter) {}

CKAttnPtr FusedRopeKVCachePrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    bool      batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes

    torch::Tensor cu_seqlens = torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    cu_seqlens.slice(0, 1, batch_size + 1) = attn_inputs.input_lengths.cumsum(0);
    cu_seqlens                             = cu_seqlens.cuda();
    torch::Tensor cu_kv_seqlens            = cu_seqlens;
    CKAttnPtr    attn_params;
    auto          params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.input_lengths.size(0));
    attn_params                = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->attn_type     = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens    = cu_seqlens;
    attn_params->cu_kv_seqlens = cu_kv_seqlens;
    attn_params->max_seq_len   = attn_inputs.input_lengths.max().item<int32_t>();
    return attn_params;
}

torch::Tensor FusedRopeKVCachePrefillOp::forward(const torch::Tensor& qkv,
                                                 FMHAType             fmha_type,
                                                 std::optional<torch_ext::KVCache> kv_cache,
                                                 const CKAttnPtr&    params) {
    // bool store_cache = params.common.kv_cache.has_value();
    auto kv_block_array                 = params->kv_block_array;
    const int     local_head_num        = attn_configs_.head_num;
    const int     local_head_num_kv     = attn_configs_.kv_head_num;
    const int     size_per_head         = attn_configs_.size_per_head;
    const int     token_num             = qkv.size(0);
    const int     batch_size            = params->cu_seqlens.size(0);
    // std::vector<torch::Tensor> qkv_split = qkv.split_with_sizes(
    //         {local_head_num * size_per_head, local_head_num_kv * size_per_head, local_head_num_kv * size_per_head},
    //         -1);
    torch::Tensor q_output = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor k_output = torch::empty({token_num, local_head_num_kv, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor v_output = torch::empty({token_num, local_head_num_kv, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (kv_cache.has_value()) {
        prefix_prompt_param.kv_block_array = kv_block_array;
        //if (params.prefix_lengths.size(0)) {
        //     prefix_prompt_param.d_prefix_prompt_lengths  = params.prefix_lengths.data_ptr<int>();
        //     prefix_prompt_param.max_prefix_prompt_length = params.prefix_lengths.max().item<int>();
        //     prefix_prompt_param.count_length             = 1;
        //}
    }

    bool store_qkv = true;
    bool store_q = false;
    bool store_kv = false;
    bool store_cache          = kv_cache.has_value();
    if (bool(autil::EnvUtil::getEnv("USE_AITER_PA", 0L))) {
        hipStream_t stream_ = device_->getStream();
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTransposePrefill,
            q_output.data_ptr(),
            k_output.data_ptr(),
            v_output.data_ptr(),
            &prefix_prompt_param,
            qkv.data_ptr(), 
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            params->cu_seqlens.data_ptr<int>(),
            batch_size,
            params->max_seq_len,
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,
            0,
            false,       // use_paged_fmha
            store_qkv,   // store_qkv
            false,       // store_q
            store_kv,    // store_kv
            store_cache, // store_cache
            device_->getStream()  // 必须作为最后一个参数
        );
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTranspose,
            nullptr,
            q_output.data_ptr(),
            k_output.data_ptr(),
            v_output.data_ptr(),
            &prefix_prompt_param,
            qkv.data_ptr(), 
            nullptr,  // qkv_buf_fp8 != nullptr ? qkv_buf_fp8->data() : nullptr,
            nullptr,  // params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size *
                      // params.configs.rope_config.index_factor): nullptr,
            nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                      // params.weights.qkv_weight->bias->data() : nullptr,
            nullptr,  // params.common.padding_offset->data<int>(),
            params->cu_seqlens.data_ptr<int>(),
            batch_size,
            params->max_seq_len,  // seq_len
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,  // scale_out_ptr,
            0,        // int8_mode,
            false,
            store_qkv,
            false,  // store_q_no_transpose
            store_q,
            store_kv,
            store_cache,
            device_->getStream());
        check_cuda_error();
    }
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<CKAttn, std::shared_ptr<CKAttn>>(m, "CKAttn").def(pybind11::init<>());
    pybind11::class_<FusedRopeKVCachePrefillOp>(m, "FusedRopeKVCachePrefillOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("prepare", &FusedRopeKVCachePrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
}
}