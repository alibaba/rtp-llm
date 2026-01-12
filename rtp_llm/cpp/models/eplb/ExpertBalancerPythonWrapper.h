#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

struct EplbPlanTensors {
    int           layer_id;
    torch::Tensor layer_id_buf;
    torch::Tensor logic_expert_cnt;
    torch::Tensor log2phy;
    torch::Tensor phy2log;
    torch::Tensor moe_weight_1;  // w1 & w3
    torch::Tensor moe_weight_2;  // w2
    torch::Tensor moe_scale_1;
    torch::Tensor moe_scale_2;

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
                moe_weight_1,
                moe_weight_2,
                moe_scale_1,
                moe_scale_2};
    }
};

class ExpertBalancerPythonWrapper {
public:
    ExpertBalancerPythonWrapper() = default;
    ExpertBalancerPythonWrapper(py::object py_eplb);

    void createBalancePlan(torch::Tensor&   log_stats,
                           torch::Tensor&   gpu_loads,
                           EplbPlanTensors& eplb_plan,
                           torch::Tensor&   active_ranks_tensor);

    void loadBalanceWeight(int ep_rank, int ep_size, EplbPlanTensors& eplb_plan);

    void updateBalanceWeight(EplbPlanTensors& eplb_plan, GptModel& model);

    void createDownScalePlan(torch::Tensor& log_stats, EplbPlanTensors& eplb_plan, torch::Tensor& active_ranks_tensor);

private:
    py::object py_eplb_;
};

}  // namespace rtp_llm