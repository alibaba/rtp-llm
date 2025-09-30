#include <torch/script.h>
#include "rtp_llm/cpp/test/ModelTestUtil.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/models/models_weight/W.h"

#include <filesystem>

using namespace std;

namespace rtp_llm {

const vector<string> global_weight_keys = {W::embedding,
                                           W::lm_head,
                                           W::prefix_w,
                                           W::pre_decoder_ln_beta,
                                           W::pre_decoder_ln_gamma,
                                           W::wpe,
                                           W::final_ln_gamma,
                                           W::final_ln_beta};

const vector<string> layer_weight_keys = {W::pre_ln_gamma,     W::pre_ln_beta,   W::pre_attn_ln_gamma,
                                          W::pre_attn_ln_beta, W::attn_qkv_w,    W::attn_qkv_b,
                                          W::attn_ln_gamma,    W::attn_ln_beta,  W::attn_o_w,
                                          W::attn_o_b,         W::post_ln_gamma, W::post_ln_beta,
                                          W::ffn_w1,           W::ffn_b1,        W::ffn_w3,
                                          W::ffn_b3,           W::ffn_ln_gamma,  W::ffn_ln_beta,
                                          W::ffn_w2,           W::ffn_b2,        W::post_ffn_ln_gamma,
                                          W::post_ffn_ln_beta};

unique_ptr<const Weights> loadWeightsFromDirViaNumpy(std::string dir_path) {
    return nullptr;
}

bool hasPtFile(std::string dir_path) {
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".pt") {
            return true;
        }
    }
    return false;
}

bool hasNpyFile(std::string dir_path) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.path().extension() == ".npy") {
            return true;
        }
    }
    return false;
}

unique_ptr<const Weights> loadWeightsFromDirViaTorchScript(std::string dir_path) {
    TensorMap  model_global_weights_;
    TensorMaps layer_weights_;
    auto       py_tensors_container = torch::jit::load(dir_path + "/pytorch_tensors.pt");
    for (const auto& key : global_weight_keys) {
        try {
            auto tensor                = py_tensors_container.attr(key).toTensor();
            model_global_weights_[key] = tensor;
            RTP_LLM_LOG_INFO("model Tensor [%s] loaded: %s", key.c_str(), tensor.toString().c_str());
        } catch (const exception& e) {
            RTP_LLM_LOG_INFO("Tensor [%s] skipped: %s", key.c_str(), e.what());
            continue;
        }
    }

    size_t max_layer_id = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if ((entry.path().extension() == ".pt") && (entry.path().stem().string().compare(0, 6, "layer_") == 0)) {
            const auto layer_id_str = entry.path().stem().string().substr(6);
            const auto layer_id     = stoi(layer_id_str);
            if (layer_id > max_layer_id) {
                max_layer_id = layer_id;
            }
        }
    }

    for (size_t i = 0; i <= max_layer_id; i++) {
        auto layer_container = torch::jit::load(dir_path + "/layer_" + to_string(i) + ".pt");
        unordered_map<string, th::Tensor> layer_weights;
        for (const auto& key : layer_weight_keys) {
            try {
                auto tensor        = layer_container.attr(key).toTensor();
                layer_weights[key] = tensor;
                RTP_LLM_LOG_INFO("layer %d Tensor [%s] loaded: %s", i, key.c_str(), tensor.toString().c_str());
            } catch (const exception& e) {
                RTP_LLM_LOG_INFO("layer %d Tensor [%s] skipped: %s", i, key.c_str(), e.what());
                continue;
            }
        }
        layer_weights_.push_back(move(layer_weights));
    }

    WeightsConverter converter(true);

    auto weights = converter.createGptWeights(std::make_unique<TensorMaps>(std::move(layer_weights_)),
                                              std::make_unique<TensorMap>(std::move(model_global_weights_)));
    return weights;
}

unique_ptr<const Weights> loadWeightsFromDir(std::string dir_path) {
    if (hasPtFile(dir_path)) {
        return loadWeightsFromDirViaTorchScript(dir_path);
    } else if (hasNpyFile(dir_path)) {
        return loadWeightsFromDirViaNumpy(dir_path);
    } else {
        throw std::runtime_error("No .pt or .npy file found in " + dir_path);
    }
}

unique_ptr<GptModel> createGptModel(const GptModelInitParams& params) {
    // TODO(yitian team): create own model implementation and return.
    if (params.device->getDeviceProperties().type == rtp_llm::DeviceType::Yitian) {
        return make_unique<GptModel>(params);
    }
    return make_unique<GptModel>(params);
}

}  // namespace rtp_llm
