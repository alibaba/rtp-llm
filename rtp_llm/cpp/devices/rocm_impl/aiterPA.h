#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "attention.h"

namespace rtp_llm {
void runAiterPA(const AttentionModuleParams& params,
		rtp_llm::DeviceBase*         device,
		Buffer&                      q_tmp);
} // namespace rtp_llm
