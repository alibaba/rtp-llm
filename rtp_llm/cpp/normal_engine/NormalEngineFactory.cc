#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodeProfile.h"

namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams&                       params,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params):
    NormalEngine(params, std::move(propose_params), createDsv4DpDecodeCoordinator) {}

}  // namespace rtp_llm
