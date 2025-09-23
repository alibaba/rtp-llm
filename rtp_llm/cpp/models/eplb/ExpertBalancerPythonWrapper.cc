#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"

namespace rtp_llm {
void EplbPlanTensors::init(int log_exp_num, int phy_exp_num) {
    layer_id_buf     = torch::zeros({1}, torch::kInt32);
    logic_expert_cnt = torch::zeros({log_exp_num}, torch::kInt32);
    log2phy          = torch::zeros({log_exp_num, phy_exp_num - log_exp_num + 1}, torch::kInt32);
    phy2log          = torch::zeros({phy_exp_num}, torch::kInt32);
    // note: no need to init moe_weight_1 and moe_weight_2
}

ExpertBalancerPythonWrapper::ExpertBalancerPythonWrapper(py::object py_eplb): py_eplb_(std::move(py_eplb)) {}

void ExpertBalancerPythonWrapper::createBalancePlan(torch::Tensor&   log_stats,
                                                    torch::Tensor&   gpu_loads,
                                                    EplbPlanTensors& eplb_plan) {
    py::gil_scoped_acquire acquire;

    auto res = py_eplb_.attr("create_balance_plan")(log_stats, gpu_loads);

    py::tuple result_tuple = res.cast<py::tuple>();

    if (result_tuple.size() != 4) {
        throw std::runtime_error("Expected 4 return values from create_balance_plan");
    }

    eplb_plan.layer_id_buf     = result_tuple[0].cast<torch::Tensor>();
    eplb_plan.logic_expert_cnt = result_tuple[1].cast<torch::Tensor>();
    eplb_plan.log2phy          = result_tuple[2].cast<torch::Tensor>();
    eplb_plan.phy2log          = result_tuple[3].cast<torch::Tensor>();
}

void ExpertBalancerPythonWrapper::loadBalanceWeight(int ep_rank, int ep_size, EplbPlanTensors& eplb_plan) {
    py::gil_scoped_acquire acquire;

    auto res = py_eplb_.attr("load_moe_weight")(eplb_plan.layer_id_buf, ep_rank, ep_size, eplb_plan.phy2log);

    py::tuple result_tuple = res.cast<py::tuple>();

    if (result_tuple.size() != 5) {
        throw std::runtime_error("Expected 5 return values from load_moe_weight");
    }

    eplb_plan.layer_id     = result_tuple[0].cast<int>();
    eplb_plan.moe_weight_1 = result_tuple[1].cast<torch::Tensor>();
    eplb_plan.moe_weight_2 = result_tuple[2].cast<torch::Tensor>();
    eplb_plan.moe_scale_1  = result_tuple[3].cast<torch::Tensor>();
    eplb_plan.moe_scale_2  = result_tuple[4].cast<torch::Tensor>();
}

}  // namespace rtp_llm