#include "maga_transformer/cpp/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/dataclass/EngineScheduleInfo.h"
#include "maga_transformer/cpp/th_op/GptInitParameterRegister.h"
#include "maga_transformer/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "maga_transformer/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "maga_transformer/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "maga_transformer/cpp/devices/DeviceFactory.h"

using namespace rtp_llm;

namespace torch_ext {
PYBIND11_MODULE(libth_transformer, m) {
    rtp_llm::registerLoadBalanceInfo(m);
    rtp_llm::registerEngineScheduleInfo(m);
    registerGptInitParameter(m);
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);
    registerDeviceOps(m);
}

}  // namespace torch_ext
