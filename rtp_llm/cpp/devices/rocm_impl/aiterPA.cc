#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace pybind11::literals;

namespace rtp_llm {

template <typename ReturnType = void, typename... Args>
ReturnType call_func_ptr(void* func_ptr, Args... args){
    auto func = reinterpret_cast<ReturnType (*)(Args...)>(func_ptr);
    return func(std::forward<Args>(args)...);
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

AiterWrapper::AiterWrapper(const DeviceInitParams& params) {
    if (!Py_IsInitialized()) {
        return;
    }
    py::gil_scoped_acquire acquire;
    aiter_module = py::module::import("aiter");
    paged_attention_rocm = aiter_module.attr("paged_attention_rocm");

    pa_gluon_aot_api = py::module_::import("aiter_meta.csrc.cpp_itfs.pa_gluon_aot.api");
    load_all_libs = pa_gluon_aot_api.attr("load_all_libs");

    use_asm_pa_ = params.use_asm_pa;
}

void AiterWrapper::runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream) {
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);

    int64_t partition_size = 256;
    int64_t mtp = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) : (params.common.decoder_max_seq_len + 1);

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
    auto key_cache             = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto value_cache           = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);
    auto block_tables          = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);

    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;
    out = out.view(query.sizes());

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    int64_t x = 16 / key_cache.element_size();
    auto kv_sizes = key_cache.sizes();
    // [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    key_cache = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
    if (use_asm_pa_) {
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});
    }

    torch::Tensor q_quant, q_scale, query_scale_gluon;
    torch::Tensor k_scale, v_scale;

    int kv_quant_mode = -1;
    std::string kv_cache_dtype = "auto";
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        k_scale = Buffer2torchTensor(params.common.kv_cache->k_scale_buffer, false);
        v_scale = Buffer2torchTensor(params.common.kv_cache->v_scale_buffer, false);
        if (k_scale.numel() > 1) {
            kv_quant_mode = 1;
            k_scale.unsqueeze_(-1);
            v_scale.unsqueeze_(-1);
        } else {
            kv_quant_mode = 0;
        }
    }

    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;
    std::optional<torch::Tensor> alibi_slopes;

    torch::Tensor output_gluon, query_gluon;
    std::vector<void *> pa_decode_gluon_ptrs;
    {
        py::gil_scoped_acquire acquire;
        std::vector<unsigned long> py_list = load_all_libs(
                    "bfloat16",         // data_type
                    head_size,          // last_dim
                    mtp,                // query_length
                    num_heads,          // num_query_heads
                    num_kv_heads,       // num_kv_heads
                    head_size,          // head_size
                    key_cache.size(3),  // kv_block_size
                    partition_size,     // context_partition_size
                    -1,                 // query_quant_mode
                    kv_quant_mode,      // kv_quant_mode
                    kv_cache_dtype,     // kv_cache_dtype
                    true         // value_transposed
                ).cast<std::vector<unsigned long>>();
        pa_decode_gluon_ptrs.reserve(py_list.size());
        for (unsigned long item : py_list) {
            pa_decode_gluon_ptrs.push_back((void *)item);
        }
    }

    if (mtp > 1) {
        auto stride_input_batch   = mtp * num_heads * head_size;
        auto stride_input_seq     = num_heads * head_size;
        auto stride_input_head    = query_group_size * head_size;

        auto stride_input_group   = head_size;
        auto stride_output_batch  = num_kv_heads * mtp * query_group_size * head_size;
        auto stride_output_merged = head_size;

        auto merged_dim_size = num_kv_heads * mtp * query_group_size;
        auto merged_block_size = next_power_of_2(merged_dim_size);
        auto block_size_last = next_power_of_2(head_size);
        auto grid_dim_0 = batch_size;
        auto grid_dim_1 = (merged_dim_size + merged_block_size - 1) / merged_block_size;;
        auto grid_dim_2 = (head_size + block_size_last - 1) / block_size_last;

        output_gluon = torch::empty({(int)batch_size, (int)num_kv_heads * (int)mtp * (int)query_group_size, (int)head_size},
                                        torch::TensorOptions().dtype(out.dtype()).device(torch::Device(torch::kCUDA)));
        query_gluon = torch::empty({(int)batch_size, (int)num_kv_heads * (int)mtp * (int)query_group_size, (int)head_size},
                                        torch::TensorOptions().dtype(query.dtype()).device(torch::Device(torch::kCUDA)));

        auto output_gluon_sizes = output_gluon.sizes();
        auto exp_sums_sizes = exp_sums.sizes();
        auto tmp_out_sizes = tmp_out.sizes();
        auto query_gluon_sizes = query_gluon.sizes();
        auto key_cache_sizes = key_cache.sizes();
        auto value_cache_sizes = value_cache.sizes();
        auto block_tables_sizes = block_tables.sizes();

        call_func_ptr<void>(pa_decode_gluon_ptrs[0],
                            query.data_ptr(),
                            query_gluon.data_ptr(),
                            batch_size, mtp, num_kv_heads, query_group_size, head_size,
                            stride_input_batch, stride_input_seq, stride_input_head, stride_input_group,
                            stride_output_batch, stride_output_merged,
                            grid_dim_0, grid_dim_1, grid_dim_2,stream);

        call_func_ptr<void>(pa_decode_gluon_ptrs[1],
                            output_gluon.data_ptr(),
                            exp_sums.data_ptr(),
                            max_logits.data_ptr(),
                            tmp_out.data_ptr(),
                            query_gluon.data_ptr(),
                            key_cache.data_ptr(),
                            value_cache.data_ptr(),
                            block_tables.data_ptr(),
                            seq_lens.data_ptr(),
                            scale,
                            nullptr,
                            k_scale.defined()? k_scale.data_ptr() : nullptr,
                            v_scale.defined()? v_scale.data_ptr() : nullptr,
                            output_gluon.stride(0), output_gluon.stride(1),
                            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
                            query_gluon.stride(0), query_gluon.stride(1),
                            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                            block_tables.stride(0),
                            0,
                            k_scale.defined()? k_scale.stride(0):0,
                            k_scale.defined()? k_scale.stride(1):0,
                            batch_size, num_kv_heads, max_num_partitions,
                            mtp, query_group_size, multi_query_group_size, head_size, stream);

        auto output_stride_input_batch = num_kv_heads * mtp * query_group_size * head_size;
        auto output_stride_input_kv_head = mtp * query_group_size * head_size;
        auto output_stride_input_seq = query_group_size * head_size;
        auto output_stride_input_group = head_size;
        auto output_stride_output_batch_seq = num_heads * head_size;
        auto output_stride_output_merged = head_size;

        auto output_merged_dim_size = num_kv_heads * query_group_size;
        auto output_merged_block_size = next_power_of_2(output_merged_dim_size);
        auto output_block_size_last = next_power_of_2(head_size);

        auto output_grid_dim_0 = batch_size * mtp;
        auto output_grid_dim_1 = (output_merged_dim_size + output_merged_block_size - 1) / output_merged_block_size;;
        auto output_grid_dim_2 = (head_size + output_block_size_last - 1) / output_block_size_last;

        call_func_ptr<void>(pa_decode_gluon_ptrs[2],
                            output_gluon.data_ptr(),
                            out.data_ptr(),
                            batch_size,
                            mtp,
                            num_kv_heads,
                            query_group_size,
                            head_size,
                            output_stride_input_batch,
                            output_stride_input_kv_head,
                            output_stride_input_seq,
                            output_stride_input_group,
                            output_stride_output_batch_seq,
                            output_stride_output_merged,
                            output_grid_dim_0,
                            output_grid_dim_1,
                            output_grid_dim_2,
                            stream);
    } else {
        call_func_ptr<void>(pa_decode_gluon_ptrs[1],
                            out.data_ptr(),
                            exp_sums.data_ptr(),
                            max_logits.data_ptr(),
                            tmp_out.data_ptr(),
                            query.data_ptr(),
                            key_cache.data_ptr(),
                            value_cache.data_ptr(),
                            block_tables.data_ptr(),
                            seq_lens.data_ptr(),
                            scale,
                            nullptr,
                            k_scale.defined()? k_scale.data_ptr() : nullptr,
                            v_scale.defined()? v_scale.data_ptr() : nullptr,
                            out.stride(0), out.stride(1),
                            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
                            query.stride(0), query.stride(1),
                            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                            block_tables.stride(0),
                            0,
                            k_scale.defined()? k_scale.stride(0):0,
                            k_scale.defined()? k_scale.stride(1):0,
                            batch_size, num_kv_heads, max_num_partitions,
                            mtp, query_group_size, multi_query_group_size, head_size, stream);
    }
}

void AiterWrapper::runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp) {
    py::gil_scoped_acquire acquire;
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);

    int64_t partition_size = 256;
    int64_t q_length = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) : (params.common.decoder_max_seq_len + 1);

    size_t max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
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
    auto key_cache    = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto value_cache  = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);
    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;

    int64_t block_size = params.configs.tokens_per_block;

    if (use_asm_pa_) {
        int64_t x = 16 / key_cache.element_size();
        auto value_sizes = value_cache.sizes();
        value_cache = value_cache.view({value_sizes[0], value_sizes[1], value_sizes[2] / x, value_sizes[3], x});
    }

    torch::Tensor q_quant, q_scale;
    torch::Tensor k_scale, v_scale;

    std::string kv_cache_dtype = "auto";
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        k_scale = Buffer2torchTensor(params.common.kv_cache->k_scale_buffer, false);
        v_scale = Buffer2torchTensor(params.common.kv_cache->v_scale_buffer, false);
    }

    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;
    std::optional<torch::Tensor> alibi_slopes;

    paged_attention_rocm(out, exp_sums, max_logits, tmp_out, q_quant.defined() ? q_quant : query,
                         key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens,
                         block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale,
                         fp8_out_scale, partition_size, q_length, q_scale);
}

inline torch::Tensor Buffer2torchTensorCustom(const Buffer& buf, std::vector<int64_t> shape, size_t offset = 0) {
    auto option =
        torch::dtype(dataTypeToTorchType(buf.type())).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);
    return torch::from_blob((void*)((char*)(buf.data()) + offset), shape, option);
}

void runAiterAsmPA(const AttentionModuleParams& params,
		rtp_llm::DeviceBase*         device,
		Buffer&                      q_tmp) {
    auto out   = Buffer2torchTensor(params.output,false);
    auto query = Buffer2torchTensor(q_tmp,false);
    
    if (q_tmp.shape().size() < 3) {
        throw std::runtime_error("aiter_paged_attention only support 3-dim input");
    } else if (q_tmp.shape().size() > 3) {
        query = query.reshape({query.size(0), query.size(1), -1});
    }

    size_t num_heads = params.configs.head_num;
    int64_t partition_size = 256;
    int64_t max_seq_len = params.common.decoder_max_seq_len + 1;

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer,false);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer,false);

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id,false);

    auto context_lens = Buffer2torchTensor(params.common.sequence_lengths,false);
    context_lens = context_lens + 1;
    
    int max_num_blocks = block_tables.size(1);
    std::optional<torch::Tensor> K_QScale = std::nullopt;
    std::optional<torch::Tensor> V_QScale = std::nullopt;
    std::optional<torch::Tensor> out_opt = out;
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        K_QScale = Buffer2torchTensor(params.common.kv_cache->k_scale_buffer,false);
        V_QScale = Buffer2torchTensor(params.common.kv_cache->v_scale_buffer,false);
        pa_fwd(query, key_cache, value_cache, block_tables, context_lens, max_num_blocks, max_seq_len, K_QScale, V_QScale, out_opt, std::nullopt, 0);
    } else {
        pa_fwd(query, key_cache, value_cache, block_tables, context_lens, max_num_blocks, max_seq_len, K_QScale, V_QScale, out_opt);
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
    int64_t partition_size = 256;
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

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);
    /*size_t v_cache_offset = params.common.kv_cache->k_cache_buffer->sizeBytes();
    auto value_cache = Buffer2torchTensorCustom(*params.common.kv_cache->k_cache_buffer,
                                               {(int64_t)params.common.kv_cache->k_cache_buffer->shape()[0],
                                                (int64_t)params.common.kv_cache->k_cache_buffer->shape()[1],
                                                (int64_t)params.common.kv_cache->k_cache_buffer->shape()[2]},
                                               v_cache_offset);*/

    int64_t num_kv_heads = params.configs.kv_head_num;
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
    return;
}

} // namespace rtp_llm
