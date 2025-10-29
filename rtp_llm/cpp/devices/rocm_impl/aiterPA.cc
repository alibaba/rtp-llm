#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/devices/rocm_impl/atrexPA.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace pybind11::literals;

namespace rtp_llm {

AiterWrapper::AiterWrapper() {
    py::gil_scoped_acquire acquire;
    aiter_module = py::module::import("aiter");
    pa_func = aiter_module.attr("paged_attn").attr("PagedAttention").attr("forward_decode");
}

torch::Tensor read_tensor_from_file(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));
    input.close();
    torch::IValue x = torch::pickle_load(bytes);
    return x.toTensor();
}

void AiterWrapper::mtp() {
    py::gil_scoped_acquire acquire;
    printf("DEBUG: in AiterWrapper::mtp\n");
    torch::Tensor query = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/query.pt");
    torch::Tensor key_cache = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/k_cache.pt");
    torch::Tensor value_cache = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/v_cache.pt");
    torch::Tensor block_tables = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/block_table.pt");
    torch::Tensor seq_lens = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/seq_lens.pt");
    
    torch::Tensor ref = read_tensor_from_file("/mnt/raid0/hangy/aiter/op_tests/play_around_mtp/data/hip_pa_ret.pt");

    // Call the Python function
    int mtp = 5;
    float scale = 1.0 / std::sqrt(128);
    py::object result = pa_func(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        128,
        "auto",
        8,
        scale,
        py::none(),
        py::none(),
        py::none(),
        "q_scale"_a = py::none(),
        "mtp"_a = mtp,
        "output_dtype"_a = torch::kBFloat16
    );
    torch::Tensor output_tensor = result.cast<torch::Tensor>();
    printf("DEBUG: in AiterWrapper::mtp done\n");
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
