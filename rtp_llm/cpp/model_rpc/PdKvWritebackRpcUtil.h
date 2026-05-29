#pragma once

#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

PdKvWritebackLaunchRequest pdKvWritebackLaunchRequestFromPB(const PdKvWritebackRequestPB& pb);

}  // namespace rtp_llm
