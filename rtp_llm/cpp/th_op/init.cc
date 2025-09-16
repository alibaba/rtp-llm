#include <torch/library.h>
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/dataclass/WorkerStatusInfo.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"
#include "rtp_llm/cpp/th_op/common/blockUtil.h"

#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

// TODO(wangyin.yx): organize these regsiter function into classified registration functions

PYBIND11_MODULE(libth_transformer, m) {
    registerKvCacheInfo(m);
    registerWorkerStatusInfo(m);
    registerEngineScheduleInfo(m);

    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);

    registerDeviceOps(m);
    registerPyOpDefs(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
