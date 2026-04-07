#include "rtp_llm/models_py/bindings/cuda/FakeBalanceExpertOp.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

FakeBalanceExpertOp::FakeBalanceExpertOp(
    int64_t expert_num, int64_t moe_k, int64_t dp_rank, int64_t dp_size, int64_t ep_size):
    expert_num_(expert_num), moe_k_(moe_k), dp_rank_(dp_rank), dp_size_(dp_size), ep_size_(ep_size) {}

void FakeBalanceExpertOp::forward(torch::Tensor expert_ids, torch::Tensor expert_scales) {
    const auto token_num      = expert_ids.sizes()[0];
    const auto top_k          = moe_k_;
    StreamType current_stream = GET_CURRENT_STREAM();

    if (expert_ids.dtype() == torch::kInt64) {
        fake_balance_expert(expert_ids.data_ptr<int64_t>(),
                            expert_scales.data_ptr<float>(),
                            dp_rank_,
                            dp_size_,
                            ep_size_,
                            expert_num_,
                            token_num * top_k,
                            current_stream);
    } else if (expert_ids.dtype() == torch::kInt32) {
        fake_balance_expert(expert_ids.data_ptr<int32_t>(),
                            expert_scales.data_ptr<float>(),
                            dp_rank_,
                            dp_size_,
                            ep_size_,
                            expert_num_,
                            token_num * top_k,
                            current_stream);
    } else {
        throw std::runtime_error("Unimplemented dtype for FakeBalanceExpertOp: "
                                 + std::string(expert_ids.dtype().name()));
    }
}

void registerFakeBalanceExpertOp(py::module& m) {
    pybind11::class_<FakeBalanceExpertOp>(m, "FakeBalanceExpertOp")
        .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t>(),
             py::arg("expert_num"),
             py::arg("moe_k"),
             py::arg("dp_rank"),
             py::arg("dp_size"),
             py::arg("ep_size"))
        .def("forward", &FakeBalanceExpertOp::forward, py::arg("expert_ids"), py::arg("expert_scales"));
}

}  // namespace rtp_llm
