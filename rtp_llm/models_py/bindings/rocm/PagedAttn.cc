
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/models_py/bindings/rocm/PagedAttn.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

PagedAttnDecodeOp::PagedAttnDecodeOp(const AttentionConfigs& attn_configs, int layer_num, int64_t block_nums, const FMHAConfig& fmha_config):
    attn_configs_(attn_configs),
    layer_num_(layer_num),
    fmha_config_(fmha_config),
    device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())),
    kv_block_offset_(layer_num * block_nums),
    use_aiter_pa_(fmha_config.use_aiter_pa) {
}

bool PagedAttnDecodeOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return true;
    // return fmha_config_.enable_paged_trt_fmha && attn_configs_.kv_cache_dtype != KvCacheDataType::INT8;
}

CKAttnPtr PagedAttnDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int batch_size = attn_inputs.sequence_lengths.size(0);

    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }

    CKAttnPtr attn_params;
    bool use_fmha_fp8 = false;
    auto      params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0), use_fmha_fp8);

    attn_params              = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->decode_plan = true;
    attn_params->attn_type   = torchDTypeToDataType(attn_inputs.dtype);
    // attn_params->cu_seqlens                = cu_seqlens;
    // attn_params->cu_kv_seqlens             = cu_kv_seqlens;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths.cuda();
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->max_seq_len               = attn_inputs.sequence_lengths.max().item<int32_t>();

    // attn_params->stream = (int64_t)device_->getStream();
    return attn_params;
}

forward_param PagedAttnDecodeOp::forward(const torch::Tensor&              qkv,
                                         FMHAType                          fmha_type,
                                         std::optional<torch_ext::KVCache> kv_cache,
                                         const CKAttnPtr&                  params) {
    auto kv_block_array            = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
    if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel()) {
        kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
    }

    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->sequence_lengths.size(0);

    if (use_aiter_pa_) {
        PrefixPromptBatchWeightsParam prefix_prompt_param;
        prefix_prompt_param.kv_block_array = kv_block_array;

        if (params->prefix_lengths.defined() && params->prefix_lengths.numel() > 0) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params->prefix_lengths.data_ptr<int>();
            prefix_prompt_param.max_prefix_prompt_length = params->prefix_lengths.max().item<int>();
            prefix_prompt_param.count_length             = 1;
        }

        size_t seq_len = 1;

        bool store_qkv   = false;
        bool store_q     = true;
        bool store_kv    = false;
        bool store_cache = kv_cache.has_value();
        // token_num, size_per_head
        auto out = torch::empty({token_num, size_per_head}, torch::TensorOptions(qkv.dtype()).device(qkv.device()));

        torch::Tensor q_output = torch::empty({batch_size, local_head_num, size_per_head},
                                              torch::TensorOptions(qkv.dtype()).device(qkv.device()));
        auto          q_tmp    = q_output;
        auto          query    = q_tmp;

        if (q_tmp.dim() < 3) {
            throw std::runtime_error("aiter_paged_attention only support 3-dim input");
        } else if (q_tmp.dim() > 3) {
            query = query.reshape({query.size(0), query.size(1), -1});
        }

        size_t  num_seqs           = q_tmp[0].item<int64_t>();
        size_t  num_heads          = attn_configs_.head_num;
        size_t  head_size          = attn_configs_.size_per_head;
        int64_t partition_size     = 256;
        int64_t max_seq_len        = params->max_seq_len;
        size_t  max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
        auto    datatype           = params->attn_type;
        // params.attn_type

        BufferPtr exp_sums_buffer = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"exp_sums"});
        auto exp_sums = Buffer2torchTensor(exp_sums_buffer, false);

        BufferPtr max_logits_buffer = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"max_logits"});
        auto max_logits = Buffer2torchTensor(max_logits_buffer, false);

        BufferPtr tmp_out_buffer = device_->allocateBuffer(
            {datatype, {num_seqs, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});
        auto tmp_out = Buffer2torchTensor(tmp_out_buffer, false);

        int64_t num_kv_heads = attn_configs_.kv_head_num;
        double  scale        = attn_configs_.softmax_extra_scale / sqrtf(attn_configs_.size_per_head * 1.0f);
        int64_t block_size   = attn_configs_.tokens_per_block;

        auto context_lens = params->sequence_lengths;
        context_lens      = context_lens + 1;

        return {out,
                exp_sums,
                max_logits,
                tmp_out,
                query,
                num_kv_heads,
                scale,
                context_lens,
                block_size,
                max_seq_len,
                partition_size};
    };
}

void registerPagedAttnDecodeOp(py::module& m) {
    py::class_<PagedAttnDecodeOp>(m, "PagedAttnDecodeOp")
        .def(py::init<const AttentionConfigs&, int, int64_t, const FMHAConfig&>(),
             py::arg("attn_configs"), py::arg("layer_num"), py::arg("block_nums"), py::arg("fmha_config"))
        .def("support", &PagedAttnDecodeOp::support, py::arg("attn_inputs"))

        .def("prepare",
             &PagedAttnDecodeOp::prepare,
             py::arg("attn_inputs"),
             "Prepare attention parameters for the forward pass")
        .def("forward",
             &PagedAttnDecodeOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"),
             "Perform the forward pass of decoder self attention");

    // Register forward_param struct
    py::class_<forward_param>(m, "forward_param")
        .def_readwrite("out", &forward_param::out)
        .def_readwrite("exp_sums", &forward_param::exp_sums)
        .def_readwrite("max_logits", &forward_param::max_logits)
        .def_readwrite("tmp_out", &forward_param::tmp_out)
        .def_readwrite("query", &forward_param::query)
        .def_readwrite("num_kv_heads", &forward_param::num_kv_heads)
        .def_readwrite("scale", &forward_param::scale)
        .def_readwrite("context_lens", &forward_param::context_lens)
        .def_readwrite("block_size", &forward_param::block_size)
        .def_readwrite("max_seq_len", &forward_param::max_seq_len)
        .def_readwrite("partition_size", &forward_param::partition_size);
}
}  // namespace rtp_llm