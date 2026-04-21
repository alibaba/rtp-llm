#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapter.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

IMooncakeTransferEngineAdapterPtr createMooncakeTransferEngineAdapter();

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
