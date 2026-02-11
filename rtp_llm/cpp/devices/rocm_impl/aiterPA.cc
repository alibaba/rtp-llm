#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/devices/rocm_impl/atrexPA.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace pybind11::literals;

namespace rtp_llm {

template <typename ReturnType = void, typename... Args>
ReturnType call_func_ptr(void* func_ptr, Args... args){
    auto func = reinterpret_cast<ReturnType (*)(Args...)>(func_ptr);
    return func(std::forward<Args>(args)...);
}

std::string get_pa_compile_dtype(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kBFloat16:
            return "__hip_bfloat16";
        case torch::kFloat16:
            return "_Float16";
        case torch::kFloat8_e4m3fnuz:
        case torch::kFloat8_e4m3fn:
            return "uint8_t";
        default:
            throw std::runtime_error("Unsupported dtype");
    }
}

static inline uint64_t next_power_of_2(uint64_t n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

static int get_cdna_version() {
    static int cdna_version = []() {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::string gcn_arch = prop.gcnArchName;
        if (gcn_arch.find("gfx950") != std::string::npos) {
            return 4;
        } else if (gcn_arch.find("gfx942") != std::string::npos) {
            return 3;
        }
        return -1;
    }();
    return cdna_version;
}

AiterWrapper::AiterWrapper(const DeviceInitParams& params) {
    use_asm_pa_ = params.use_asm_pa;
    if (!Py_IsInitialized()) {
        return;
    }
    py::gil_scoped_acquire acquire;
    pa_gluon_aot_api = py::module_::import("aiter_meta.csrc.cpp_itfs.pa_gluon_aot.api");
    pa_gluon_load_libs = pa_gluon_aot_api.attr("load_all_libs");
    hip_pa_api = py::module_::import("aiter_meta.csrc.cpp_itfs.pa.pa_api");
    hip_pa_load_libs = hip_pa_api.attr("load_all_libs");
}

void AiterWrapper::runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream) {
    py::gil_scoped_acquire acquire;
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);
    int64_t partition_size = 256;
    int64_t mtp = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) :
                (device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1);

    size_t query_group_size = num_heads / (size_t)num_kv_heads;
    size_t multi_query_group_size = num_heads / (size_t)num_kv_heads * (size_t)mtp;
    size_t max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;

    size_t batch_size = prefill_pa ? params.common.context_batch_size : params.common.decoder_batch_size;

    BufferPtr exp_sums_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size}, AllocationType::DEVICE}, {"exp_sums"});
    BufferPtr max_logits_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size}, AllocationType::DEVICE}, {"max_logits"});
    BufferPtr tmp_out_buffer = device->allocateBuffer({params.output.type(),
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size, head_size}, AllocationType::DEVICE}, {"tmp_out"});

    auto out                   = Buffer2torchTensor(params.output, false);
    auto exp_sums              = Buffer2torchTensor(exp_sums_buffer, false);
    auto max_logits            = Buffer2torchTensor(max_logits_buffer, false);
    auto tmp_out               = Buffer2torchTensor(tmp_out_buffer, false);
    auto query                 = Buffer2torchTensor(q_mtp, false);
    auto key_cache             = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache           = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);
    auto block_tables          = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);

    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    int64_t x = 16 / key_cache.element_size();
    auto kv_sizes = key_cache.sizes();
    // k_cache [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    key_cache = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
    bool value_transposed = use_asm_pa_;
    if (use_asm_pa_) {
        // v_cache [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});
    } else {
        // v_cache [num_blocks, num_kv_heads, head_size, kv_block_size]
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2]});
    }

    torch::Tensor q_quant, q_scale;
    torch::Tensor k_scale, v_scale;

    int kv_quant_mode = -1;
    std::string kv_cache_dtype = "auto";
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        k_scale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        v_scale = k_scale;
        if (k_scale.numel() > 1) {
            kv_quant_mode = 1;
            k_scale.unsqueeze_(-1);
            v_scale.unsqueeze_(-1);
        } else {
            kv_quant_mode = 0;
        }
    }

    auto query_5d = query.view({(int64_t)batch_size, mtp, (int64_t)num_kv_heads, (int64_t)query_group_size, (int64_t)head_size});
    auto out_5d = out.view({(int64_t)batch_size, mtp, (int64_t)num_kv_heads, (int64_t)query_group_size, (int64_t)head_size});

    void* pa_decode_gluon_ptr = nullptr;
    {
        py::gil_scoped_acquire acquire;
        pa_decode_gluon_ptr = (void*)pa_gluon_load_libs(
                    "bfloat16",         // compute_type
                    mtp,                // query_length
                    num_heads,          // num_query_heads
                    num_kv_heads,       // num_kv_heads
                    head_size,          // head_size
                    key_cache.size(3),  // kv_block_size
                    partition_size,     // context_partition_size
                    -1,                 // query_quant_mode
                    kv_quant_mode,      // kv_quant_mode
                    kv_cache_dtype,     // kv_cache_dtype
                    value_transposed,   // value_transposed
                    0,                  // use_sinks
                    get_cdna_version()  // cdna_version
                ).cast<unsigned long>();
    }

    call_func_ptr<void>(pa_decode_gluon_ptr,
                        out_5d.data_ptr(), //[batch_size, query_length, num_kv_heads, query_group_size, head_size]
                        exp_sums.data_ptr(),
                        max_logits.data_ptr(),
                        tmp_out.data_ptr(),
                        query_5d.data_ptr(), //[batch_size, query_length, num_kv_heads, query_group_size, head_size]
                        key_cache.data_ptr(),
                        value_cache.data_ptr(),
                        block_tables.data_ptr(),
                        seq_lens.data_ptr(),
                        nullptr, // sinks_ptr
                        scale,
                        nullptr, // query_scale
                        k_scale.defined() ? k_scale.data_ptr() : nullptr,
                        v_scale.defined() ? v_scale.data_ptr() : nullptr,
                        out_5d.stride(0),
                        out_5d.stride(1),
                        out_5d.stride(2),
                        out_5d.stride(3),
                        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                        tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
                        query_5d.stride(0),
                        query_5d.stride(1),
                        query_5d.stride(2),
                        query_5d.stride(3),
                        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                        block_tables.stride(0),
                        0, // stride_query_scale_bs
                        0, // stride_query_scale_qlen
                        0, // stride_query_scale_kv_head
                        k_scale.defined() ? k_scale.stride(0) : 0,
                        k_scale.defined() ? k_scale.stride(1) : 0,
                        batch_size, num_kv_heads, max_num_partitions,
                        head_size, stream);
}

void AiterWrapper::runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp, hipStream_t stream) {
    py::gil_scoped_acquire acquire;
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);

    int64_t warp_size = 64;
    int64_t partition_size = 256;
    int64_t q_length = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) :
                (device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1);

    size_t query_group_size = num_heads / (size_t)num_kv_heads;
    size_t max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
    size_t npar_loops = (max_num_partitions + warp_size - 1) / warp_size;
    size_t batch_size = prefill_pa ? params.common.context_batch_size : params.common.decoder_batch_size;

    BufferPtr exp_sums_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {token_num, num_heads, max_num_partitions}, AllocationType::DEVICE}, {"exp_sums"});

    BufferPtr max_logits_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {token_num, num_heads, max_num_partitions}, AllocationType::DEVICE}, {"max_logits"});

    BufferPtr tmp_out_buffer = device->allocateBuffer({params.output.type(),
            {token_num, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});

    auto out          = Buffer2torchTensor(params.output, false);
    auto exp_sums     = Buffer2torchTensor(exp_sums_buffer, false);
    auto max_logits   = Buffer2torchTensor(max_logits_buffer, false);
    auto tmp_out      = Buffer2torchTensor(tmp_out_buffer, false);
    auto query        = Buffer2torchTensor(q_tmp, false);
    auto key_cache    = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache  = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);
    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);
    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;

    size_t max_num_blocks = block_tables.size(1);
    int64_t block_size = params.configs.tokens_per_block;

    bool v_shuffle = use_asm_pa_;
    int64_t x = 16 / key_cache.element_size();
    auto kv_sizes = value_cache.sizes();
    key_cache = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
    if (use_asm_pa_) {
        // v_cache [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});
    } else {
        // v_cache [num_blocks, num_kv_heads, head_size, kv_block_size]
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2]});
    }

    torch::Tensor q_quant, q_scale;
    torch::Tensor k_scale, v_scale;

    std::string kv_cache_dtype = "auto";
    std::string quant_method = "vllm::Fp8QuantMethod::kPerTensor";

    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        quant_method = "vllm::Fp8QuantMethod::kPerHead";
        k_scale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        v_scale = k_scale;
    }

    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;
    std::optional<torch::Tensor> alibi_slopes;

    void* paged_attention_rocm_ptr;
    {
        py::gil_scoped_acquire acquire;
        paged_attention_rocm_ptr = (void*)hip_pa_load_libs(
            query_group_size,
            head_size,
            npar_loops,
            get_pa_compile_dtype(query.scalar_type()),
            get_pa_compile_dtype(key_cache.scalar_type()),
            kv_cache_dtype,
            get_pa_compile_dtype(out.scalar_type()),
            block_size,
            alibi_slopes.has_value() ? "true" : "false",
            q_length, //query_length
            quant_method,
            v_shuffle).cast<unsigned long>();
    }

    call_func_ptr<void>(paged_attention_rocm_ptr,
                        out.data_ptr(),
                        exp_sums.data_ptr(),
                        max_logits.data_ptr(),
                        tmp_out.data_ptr(),
                        query.data_ptr(),
                        key_cache.data_ptr(),
                        value_cache.data_ptr(),
                        scale,
                        block_tables.data_ptr(),
                        seq_lens.data_ptr(),
                        max_seq_len,
                        batch_size,
                        num_kv_heads,
                        num_heads,
                        max_num_blocks,
                        query.stride(0),
                        key_cache.stride(0),
                        key_cache.stride(1),
                        nullptr,        //alibi_slopes
                        q_scale.defined() ? q_scale.data_ptr() : nullptr,
                        k_scale.defined() ? k_scale.data_ptr() : nullptr,
                        v_scale.defined() ? v_scale.data_ptr() : nullptr,
                        nullptr,        //fp8_out_scale_ptr
                        stream);

}

inline torch::Tensor Buffer2torchTensorCustom(const Buffer& buf, std::vector<int64_t> shape, size_t offset = 0) {
    auto option =
        torch::dtype(dataTypeToTorchType(buf.type())).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);
    return torch::from_blob((void*)((char*)(buf.data()) + offset), shape, option);
}

void runAiterAsmPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp) {
    auto out   = Buffer2torchTensor(params.output, false);
    auto query = Buffer2torchTensor(q_tmp, false);

    if (q_tmp.shape().size() < 3) {
        throw std::runtime_error("aiter_paged_attention only support 3-dim input");
    } else if (q_tmp.shape().size() > 3) {
        query = query.reshape({query.size(0), query.size(1), -1});
    }

    size_t  num_heads      = params.configs.head_num;
    int64_t partition_size = 256;
    int64_t max_seq_len    = params.common.decoder_max_seq_len + 1;

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);

    auto context_lens = Buffer2torchTensor(params.common.sequence_lengths, false);
    context_lens      = context_lens + 1;

    int                          max_num_blocks = block_tables.size(1);
    std::optional<torch::Tensor> K_QScale       = std::nullopt;
    std::optional<torch::Tensor> V_QScale       = std::nullopt;
    std::optional<torch::Tensor> out_opt        = out;
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        K_QScale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        V_QScale = K_QScale;
        pa_fwd(query,
               key_cache,
               value_cache,
               block_tables,
               context_lens,
               max_num_blocks,
               max_seq_len,
               K_QScale,
               V_QScale,
               out_opt,
               std::nullopt,
               0);
    } else {
        pa_fwd(query,
               key_cache,
               value_cache,
               block_tables,
               context_lens,
               max_num_blocks,
               max_seq_len,
               K_QScale,
               V_QScale,
               out_opt);
    }
}

void runAiterPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp) {
    auto out   = Buffer2torchTensor(params.output, false);
    auto query = Buffer2torchTensor(q_tmp, false);

    if (q_tmp.shape().size() < 3) {
        throw std::runtime_error("aiter_paged_attention only support 3-dim input");
    } else if (q_tmp.shape().size() > 3) {
        query = query.reshape({query.size(0), query.size(1), -1});
    }

    size_t  num_seqs       = q_tmp.shape()[0];
    size_t  num_heads      = params.configs.head_num;
    size_t  head_size      = params.configs.size_per_head;
    int64_t partition_size = 512;
    int64_t max_seq_len =
        device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1;
    size_t    max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
    auto      datatype           = params.output.type();
    BufferPtr exp_sums_buffer    = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
        {"exp_sums"});
    auto exp_sums = Buffer2torchTensor(exp_sums_buffer, false);

    BufferPtr max_logits_buffer = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
        {"max_logits"});
    auto max_logits = Buffer2torchTensor(max_logits_buffer, false);

    BufferPtr tmp_out_buffer = device->allocateBuffer(
        {datatype, {num_seqs, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});
    auto tmp_out = Buffer2torchTensor(tmp_out_buffer, false);

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);
    /*size_t v_cache_offset = params.common.kv_cache->kv_cache_buffer->sizeBytes();
    auto value_cache = Buffer2torchTensorCustom(*params.common.kv_cache->kv_cache_buffer,
                                               {(int64_t)params.common.kv_cache->kv_cache_buffer->shape()[0],
                                                (int64_t)params.common.kv_cache->kv_cache_buffer->shape()[1],
                                                (int64_t)params.common.kv_cache->kv_cache_buffer->shape()[2]},
                                               v_cache_offset);*/

    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t grp_size     = num_heads / num_kv_heads;
    double  scale        = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    int64_t block_size = params.configs.tokens_per_block;

    std::string kv_cache_dtype = key_cache.dtype() == at::kFloat8_e4m3fnuz ? "fp8" : "auto";

    double k_scale = 1.0;
    double v_scale = 1.0;

    std::optional<torch::Tensor> fp8_out_scale;
    std::optional<torch::Tensor> alibi_slopes;

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);
    // int64_t max_num_blocks_per_seq = (int64_t)params.common.kv_cache->kv_cache_block_id->shape()[1];
    // auto block_tables = Buffer2torchTensorCustom(*params.common.kv_cache->kv_cache_block_id,
    //                                             {(int64_t)params.common.kv_cache->kv_cache_block_id->shape()[0],
    //                                              max_num_blocks_per_seq,
    //                                             }, 0);

    auto aiter_attn = (AiterAttnParams*)params.common.decode_aiter_attn.get();
    if (!aiter_attn) {
        throw std::runtime_error("aiter_attn must be setting when using aiter pa");
    }

    auto context_lens = aiter_attn->sequence_lengths_t;
    if (max_seq_len <= 16384) {
        int64_t x        = 16 / key_cache.element_size();
        auto    kv_sizes = value_cache.sizes();
        out              = out.view({int64_t(num_seqs), int64_t(num_heads), int64_t(head_size)});
        exp_sums   = exp_sums.view({int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size});
        max_logits = max_logits.view({int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size});
        tmp_out    = tmp_out.view(
            {int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size, int64_t(head_size)});
        query       = query.view({int64_t(num_seqs), int64_t(num_heads), int64_t(head_size)});
        key_cache   = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2]});
        paged_attention_atrex(out,
                              exp_sums,
                              max_logits,
                              tmp_out,
                              query,
                              key_cache,
                              value_cache,
                              context_lens,
                              block_tables,
                              scale,
                              max_seq_len,
                              alibi_slopes);
    } else {
        partition_size = 256;
        max_seq_len =
            device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1;
        max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
        datatype           = params.output.type();
        exp_sums_buffer    = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"exp_sums"});
        exp_sums = Buffer2torchTensor(exp_sums_buffer, false);

        max_logits_buffer = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"max_logits"});
        max_logits = Buffer2torchTensor(max_logits_buffer, false);

        tmp_out_buffer = device->allocateBuffer(
            {datatype, {num_seqs, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});
        tmp_out = Buffer2torchTensor(tmp_out_buffer, false);
        paged_attention(out,
                        exp_sums,
                        max_logits,
                        tmp_out,
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        context_lens,
                        block_size,
                        max_seq_len,
                        alibi_slopes,
                        kv_cache_dtype,
                        k_scale,
                        v_scale,
                        fp8_out_scale,
                        partition_size);
    }
    return;
}

}  // namespace rtp_llm
