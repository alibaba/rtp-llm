#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

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
