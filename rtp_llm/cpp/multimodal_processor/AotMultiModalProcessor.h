#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace rtp_llm {

class AotMultimodalProcessor: public MultimodalProcessor {
public:
    AotMultimodalProcessor(py::object mm_process_engine, rtp_llm::GptInitParameter params):
        MultimodalProcessor(mm_process_engine, params) {
        std::string root_path = autil::EnvUtil::getEnv("HIPPO_APP_INST_ROOT", "");
        if (root_path.empty()) {
            root_path = params.ckpt_path_;
        }
        std::string model_file_path = root_path + "/custom_modal/custom_modal.so";
        std::string aot_model_path  = root_path + "/custom_modal";
        RTP_LLM_LOG_INFO("AotMultimodalProcessor load model from %s", model_file_path.c_str());
        _aot_model_container_runner = std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(
            model_file_path, _aot_model_parallel_num, "cuda", aot_model_path);
    }

private:
    std::shared_ptr<torch::inductor::AOTIModelContainerRunner> _aot_model_container_runner;
    int                                                        _aot_model_parallel_num = 24;

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        MultimodalOutput           mm_embedding_res;
        std::vector<torch::Tensor> mm_features;
        for (const auto& mm_input : mm_inputs) {
            std::vector<at::Tensor> cuda_inputs;
            cuda_inputs.reserve(mm_input.tensors.size());
            for (const auto& t : mm_input.tensors) {
                cuda_inputs.push_back(t.to(torch::kCUDA, /* non_blocking = */ true));
            }
            std::vector<at::Tensor> outputs = _aot_model_container_runner->run(cuda_inputs);
            assert(outputs.size() == 1);
            mm_features.push_back(outputs[0]);
        }
        mm_embedding_res.mm_features = mm_features;

        return mm_embedding_res;
    }
};

}  // namespace rtp_llm
