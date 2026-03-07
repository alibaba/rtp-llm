#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

struct EplbPlanTensors {
    int           layer_id;
    torch::Tensor layer_id_buf;
    torch::Tensor logic_expert_cnt;
    torch::Tensor log2phy;
    torch::Tensor phy2log;
    torch::Tensor moe_gate_up_weight;  // gate & up (fused)
    torch::Tensor moe_down_weight;     // down
    torch::Tensor moe_gate_up_scale;
    torch::Tensor moe_down_scale;

    void init(int log_exp_num, int phy_exp_num);

    using tuple_type = std::tuple<int,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor>;

    tuple_type to_tuple() const {
        return {layer_id,
                layer_id_buf,
                logic_expert_cnt,
                log2phy,
                phy2log,
                moe_gate_up_weight,
                moe_down_weight,
                moe_gate_up_scale,
                moe_down_scale};
    }
};

class ExpertBalancerPythonWrapper {
public:
    ExpertBalancerPythonWrapper() = default;
    ExpertBalancerPythonWrapper(py::object py_eplb);

    void createBalancePlan(torch::Tensor& log_stats, torch::Tensor& gpu_loads, EplbPlanTensors& eplb_plan);

    void loadBalanceWeight(int ep_rank, int ep_size, EplbPlanTensors& eplb_plan);

private:
    py::object py_eplb_;
};

}  // namespace rtp_llm