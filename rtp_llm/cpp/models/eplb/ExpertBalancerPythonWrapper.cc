#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

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
                                                    EplbPlanTensors& eplb_plan,
                                                    torch::Tensor&   active_ranks_tensor) {
    py::gil_scoped_acquire acquire;

    auto res = py_eplb_.attr("create_balance_plan")(log_stats, gpu_loads, active_ranks_tensor);

    py::tuple result_tuple = res.cast<py::tuple>();

    if (result_tuple.size() != 4) {
        throw std::runtime_error("Expected 4 return values from create_balance_plan");
    }

    eplb_plan.layer_id_buf     = result_tuple[0].cast<torch::Tensor>();
    eplb_plan.logic_expert_cnt = result_tuple[1].cast<torch::Tensor>();
    eplb_plan.log2phy          = result_tuple[2].cast<torch::Tensor>();
    eplb_plan.phy2log          = result_tuple[3].cast<torch::Tensor>();
}

void ExpertBalancerPythonWrapper::createDownScalePlan(torch::Tensor&   log_stats,
                                                      EplbPlanTensors& eplb_plan,
                                                      torch::Tensor&   active_ranks_tensor) {
    py::gil_scoped_acquire acquire;

    auto res = py_eplb_.attr("create_downscale_plan")(log_stats, active_ranks_tensor);

    py::tuple result_tuple = res.cast<py::tuple>();

    if (result_tuple.size() != 3) {
        throw std::runtime_error("Expected 3 return values from create_downscale_plan");
    }

    eplb_plan.logic_expert_cnt = result_tuple[0].cast<torch::Tensor>();
    eplb_plan.log2phy          = result_tuple[1].cast<torch::Tensor>();
    eplb_plan.phy2log          = result_tuple[2].cast<torch::Tensor>();
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

void ExpertBalancerPythonWrapper::updateBalanceWeight(EplbPlanTensors& eplb_plan, GptModel& model) {
    py::gil_scoped_acquire acquire;
    // Try to get Python model from PyWrappedModel
    PyWrappedModel* py_model = dynamic_cast<PyWrappedModel*>(&model);
    if (py_model == nullptr) {
        // Not a PyWrappedModel, skip Python weight update
        return;
    }

    try {
        py::object py_model_obj = py_model->getPyModel();
        if (py_model_obj.is_none() || !py::hasattr(py_model_obj, "weight")) {
            // Python model or weight not available
            return;
        }

        py::object py_weights = py_model_obj.attr("weight");

        // Update weights using ModelWeights.update_layer_weight
        // Note: weight names should match W.moe_w1, W.moe_w2, etc.
        py_weights.attr("update_layer_weight")(
            eplb_plan.layer_id, "partial_moe_weights.intermediate_weight.kernel", eplb_plan.moe_weight_1.cuda());
        py_weights.attr("update_layer_weight")(
            eplb_plan.layer_id, "partial_moe_weights.intermediate_weight2.kernel", eplb_plan.moe_weight_2.cuda());

        // Update scales if they exist and are not empty
        if (eplb_plan.moe_scale_1.defined() && eplb_plan.moe_scale_1.numel() > 0) {
            py_weights.attr("update_layer_weight")(eplb_plan.layer_id,
                                                   "partial_moe_weights.intermediate_weight.weight_only_quant_scale",
                                                   eplb_plan.moe_scale_1.cuda());
        }
        if (eplb_plan.moe_scale_2.defined() && eplb_plan.moe_scale_2.numel() > 0) {
            py_weights.attr("update_layer_weight")(eplb_plan.layer_id,
                                                   "partial_moe_weights.intermediate_weight2.weight_only_quant_scale",
                                                   eplb_plan.moe_scale_2.cuda());
        }

        // Update EPLB mapping weights
        if (eplb_plan.log2phy.defined() && eplb_plan.log2phy.numel() > 0) {
            py_weights.attr("update_layer_weight")(eplb_plan.layer_id, "moe_eplb.log2phy", eplb_plan.log2phy.cuda());
        }
        if (eplb_plan.logic_expert_cnt.defined() && eplb_plan.logic_expert_cnt.numel() > 0) {
            py_weights.attr("update_layer_weight")(
                eplb_plan.layer_id, "moe_eplb.logic_expert_cnt", eplb_plan.logic_expert_cnt.cuda());
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to update Python weights after EPLB: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to update Python weights after EPLB: %s", e.what());
    }
}

}  // namespace rtp_llm