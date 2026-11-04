#include "rtp_llm/flexlb/navi_sched/service/NaviSchedRoleSummary.h"

#include <sstream>

namespace navi_sched {

void NaviSchedRoleSummary::Jsonize(JsonWrapper &json)
{
    json.Jsonize("name", name);
    json.Jsonize("role", role);
    json.Jsonize("started", started);
    json.Jsonize("grpc_port", grpcPort);
}

std::string NaviSchedRoleSummary::format() const
{
    std::ostringstream output;
    output << "Role\n"
           << "  name: " << name << "\n"
           << "  type: " << role << "\n"
           << "  started: " << (started ? "true" : "false") << "\n"
           << "  grpc_port: " << grpcPort;
    return output.str();
}

} // namespace navi_sched
