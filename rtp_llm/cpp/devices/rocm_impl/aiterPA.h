#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
// #include "aiter_meta/csrc/include/attention.h"
#include "attention.h"
#include "attention_asm.h"

namespace rtp_llm {
void runAiterAsmPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
void runAiterPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
}  // namespace rtp_llm
