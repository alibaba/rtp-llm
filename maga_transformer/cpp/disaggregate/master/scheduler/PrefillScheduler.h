#pragma once
#include "maga_transformer/cpp/disaggregate/master/scheduler/Struct.h"
#include "maga_transformer/cpp/disaggregate/master/cluster/PrefillCluster.h"
#include "absl/status/statusor.h"

namespace rtp_llm {
namespace rtp_llm_master {

class PrefillScheduler {
public:
    absl::StatusOr<MachineInfo> scheduleRequest(RequestInfo);
private:
    PrefillCluster cluster;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm