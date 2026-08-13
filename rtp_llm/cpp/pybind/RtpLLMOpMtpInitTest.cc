#include <memory>

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

TEST(RtpLLMOpMtpInitTest, RecurrentProposalKeepsAllLayersInOneModelId) {
    py::scoped_interpreter interpreter;

    {
        auto types       = py::module_::import("types");
        auto test_module = types.attr("ModuleType")("mtp_init_test_config");
        py::class_<ModelConfig>(test_module, "ModelConfig").def(py::init<>());
        py::enum_<SpeculativeType>(test_module, "SpeculativeType").value("MTP", SP_TYPE_MTP);
        py::module_::import("torch");

        ModelConfig draft_config;
        draft_config.num_layers = 2;

        py::list layer_weights;
        for (float marker : {11.0F, 22.0F}) {
            py::dict layer;
            layer["pre_layernorm_weights.gamma"] = py::cast(torch::tensor({marker}, torch::kFloat32));
            layer_weights.append(std::move(layer));
        }

        auto weight = types.attr("SimpleNamespace")();
        weight.attr("weights")        = std::move(layer_weights);
        weight.attr("global_weights") = py::dict();

        auto draft_model = types.attr("SimpleNamespace")();
        draft_model.attr("model_config") = py::cast(draft_config);
        draft_model.attr("weight")       = std::move(weight);

        auto proposal = types.attr("SimpleNamespace")();
        proposal.attr("model")   = std::move(draft_model);
        proposal.attr("sp_type") = py::cast(SP_TYPE_MTP);

        EngineInitParams base_params;
        base_params.sp_config.gen_num_per_cycle = 3;

        RtpLLMOp op;
        op.model_id_ = 41;
        auto proposal_params = op.initProposeModel(proposal, base_params);

        ASSERT_NE(proposal_params, nullptr);
        ASSERT_NE(proposal_params->mtp_model_params_, nullptr);
        ASSERT_EQ(proposal_params->mtp_model_params_->size(), 1u);

        const auto& draft_params = *proposal_params->mtp_model_params_->front();
        EXPECT_EQ(draft_params.model_id, 41u);
        EXPECT_EQ(draft_params.model_config_.num_layers, 2);
        ASSERT_EQ(draft_params.gpt_weights.layers.size(), 2u);
        ASSERT_NE(draft_params.gpt_weights.layers[0].pre_layernorm, nullptr);
        ASSERT_NE(draft_params.gpt_weights.layers[1].pre_layernorm, nullptr);
        EXPECT_FLOAT_EQ(draft_params.gpt_weights.layers[0].pre_layernorm->gamma.item<float>(), 11.0F);
        EXPECT_FLOAT_EQ(draft_params.gpt_weights.layers[1].pre_layernorm->gamma.item<float>(), 22.0F);
        EXPECT_EQ(op.model_id_, 42u);

        base_params.sp_config.gen_num_per_cycle = 0;
        EXPECT_ANY_THROW(op.initProposeModel(proposal, base_params));
        base_params.sp_config.gen_num_per_cycle = -1;
        EXPECT_ANY_THROW(op.initProposeModel(proposal, base_params));
        EXPECT_EQ(op.model_id_, 42u);

        base_params.sp_config.gen_num_per_cycle = 3;
        auto all_weights = proposal.attr("model").attr("weight").attr("weights").cast<py::list>();
        py::list one_layer;
        one_layer.append(all_weights[0]);
        proposal.attr("model").attr("weight").attr("weights") = std::move(one_layer);
        EXPECT_ANY_THROW(op.initProposeModel(proposal, base_params));

        proposal.attr("model").attr("weight").attr("weights") = py::list();
        EXPECT_ANY_THROW(op.initProposeModel(proposal, base_params));
        EXPECT_EQ(op.model_id_, 42u);

        auto make_draft_params = [](int64_t config_layers, size_t weight_layers) {
            auto params                      = std::make_unique<EngineInitParams>();
            params->model_config_.num_layers = config_layers;
            params->gpt_weights.layers.resize(weight_layers);
            return params;
        };

        auto duplicate_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        duplicate_params->push_back(make_draft_params(1, 1));
        duplicate_params->push_back(make_draft_params(1, 1));
        EXPECT_ANY_THROW({
            ProposeModelEngineInitParams invalid(SP_TYPE_MTP, 3, std::move(duplicate_params));
        });

        auto mismatched_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        mismatched_params->push_back(make_draft_params(2, 1));
        EXPECT_ANY_THROW({
            ProposeModelEngineInitParams invalid(SP_TYPE_MTP, 3, std::move(mismatched_params));
        });
    }
}

}  // namespace
}  // namespace rtp_llm
