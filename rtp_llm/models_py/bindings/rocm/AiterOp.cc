#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/rocm/AiterOp.h"

namespace rtp_llm {

AiterAttnPyParams::AiterAttnPyParams() = default;

AiterAttnPyParams::AiterAttnPyParams(torch::Tensor input_lengths, bool is_prefill): is_prefill_(is_prefill) {
    // 使用基类成员变量
    input_lengths_ = input_lengths;
    batch_size_    = input_lengths.size(0);
    max_seq_len_   = input_lengths.max().item<int>();
    if (is_prefill_) {
        cu_seqlens_q_ =
            torch::zeros({batch_size_ + 1}, torch::TensorOptions().dtype(torch::kInt32).device(input_lengths.device()));
        cu_seqlens_q_.slice(0, 1, batch_size_ + 1) = torch::cumsum(input_lengths_, 0);
        cu_seqlens_k_                              = cu_seqlens_q_.clone();
        max_seqlen_q_                              = max_seq_len_;
        max_seqlen_k_                              = max_seq_len_;
    }
}

AiterAttnPyParams::AiterAttnPyParams(torch::Tensor input_lengths,
                                     torch::Tensor sequence_lengths,
                                     torch::Tensor kv_cache_block_id_host,
                                     torch::Tensor kv_cache_block_id_device,
                                     bool          enable_cuda_graph):
    is_prefill_(false), enable_cuda_graph_(enable_cuda_graph) {
    input_lengths_            = input_lengths;     // CPU or CUDA都可
    sequence_lengths_         = sequence_lengths;  // 用户输入的CPU buffer
    kv_cache_block_id_host_   = kv_cache_block_id_host;
    kv_cache_block_id_device_ = kv_cache_block_id_device;

    batch_size_ = input_lengths.size(0);
    if (enable_cuda_graph) {
        max_seq_len_ = 8192;
    } else {
        max_seq_len_ = input_lengths.max().item<int>() + 1;
    }
    max_seqlen_k_ = max_seq_len_;
    seq_lens_     = (sequence_lengths_ + 1).to(torch::kCUDA);
}

void AiterAttnPyParams::update() {
    if (seq_lens_.defined()) {
        seq_lens_.copy_((sequence_lengths_ + 1).to(torch::kCUDA));
    } else {
        assert(false && "seq_lens_ not created");
    }
    max_seq_len_ = 8192;
}

bool AiterAttnPyParams::check_recycle() {
    return true;
}

}  // namespace rtp_llm

namespace rtp_llm {

// 新的参数创建器类
class AiterParamsCreator {
public:
    AiterParamsCreator() = default;

    // 创建 prefill 参数
    ParamsBasePtr create_prefill_params(std::shared_ptr<AiterAttnPyParams> params) {
        return ParamsBasePtr(params);
    }

    // 创建 decode 参数
    ParamsBasePtr create_decode_params(std::shared_ptr<AiterAttnPyParams> params) {
        return ParamsBasePtr(params);
    }
};

void registerAiterOp(const pybind11::module& m) {
    pybind11::class_<rtp_llm::AiterAttnPyParams, std::shared_ptr<rtp_llm::AiterAttnPyParams>, rtp_llm::ParamsBase>(
        m, "AiterAttnPyParams")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, bool>(), pybind11::arg("input_lengths"), pybind11::arg("is_prefill"))
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool>(),
             pybind11::arg("input_lengths"),
             pybind11::arg("sequence_lengths"),
             pybind11::arg("kv_cache_block_id_host"),
             pybind11::arg("kv_cache_block_id_device"),
             pybind11::arg("enable_cuda_graph"))
        .def("fill_params",
             [](rtp_llm::AiterAttnPyParams& self,
                torch::Tensor               sequence_lengths,
                torch::Tensor               input_lengths,
                torch::Tensor               kv_cache_block_id_host,
                int                         batch_size,
                int                         seq_size_per_block) {
                 self.fillParams(
                     sequence_lengths, input_lengths, kv_cache_block_id_host, batch_size, seq_size_per_block);
             })
        // 直接暴露成员变量
        .def_readwrite("is_prefill_", &rtp_llm::AiterAttnPyParams::is_prefill_)
        .def_readwrite("batch_size_", &rtp_llm::AiterAttnPyParams::batch_size_)
        .def_readwrite("max_seq_len_", &rtp_llm::AiterAttnPyParams::max_seq_len_)
        .def_readwrite("max_seqlen_q_", &rtp_llm::AiterAttnPyParams::max_seqlen_q_)
        .def_readwrite("max_seqlen_k_", &rtp_llm::AiterAttnPyParams::max_seqlen_k_)
        .def_readwrite("input_lengths_", &rtp_llm::AiterAttnPyParams::input_lengths_)
        .def_readwrite("sequence_lengths_", &rtp_llm::AiterAttnPyParams::sequence_lengths_)
        .def_readwrite("cu_seqlens_q_", &rtp_llm::AiterAttnPyParams::cu_seqlens_q_)
        .def_readwrite("cu_seqlens_k_", &rtp_llm::AiterAttnPyParams::cu_seqlens_k_)
        .def_readwrite("seq_lens_", &rtp_llm::AiterAttnPyParams::seq_lens_)
        .def_readwrite("kv_cache_block_id_host_", &rtp_llm::AiterAttnPyParams::kv_cache_block_id_host_)
        .def_readwrite("kv_cache_block_id_device_", &rtp_llm::AiterAttnPyParams::kv_cache_block_id_device_)
        .def_readwrite("enable_cuda_graph_", &rtp_llm::AiterAttnPyParams::enable_cuda_graph_);

    // 注册 AiterParamsCreator 类
    pybind11::class_<rtp_llm::AiterParamsCreator>(m, "AiterParamsCreator")
        .def(pybind11::init<>())
        .def("create_prefill_params", &rtp_llm::AiterParamsCreator::create_prefill_params, pybind11::arg("params"))
        .def("create_decode_params", &rtp_llm::AiterParamsCreator::create_decode_params, pybind11::arg("params"));
}
}  // namespace rtp_llm
