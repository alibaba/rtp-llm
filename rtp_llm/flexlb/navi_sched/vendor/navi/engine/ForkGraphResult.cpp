#include "navi/engine/ForkGraphResult.h"

namespace navi {

void ForkGraphResult::markThisLocation(
    SingleResultLocationDef *singleLocation) const
{
    singleLocation->set_type(NRT_FORK);
    auto forkLocation = singleLocation->mutable_fork_location();
    forkLocation->set_graph_id(_graphId);
    forkLocation->set_biz_name(std::string(_bizName.c_str(), _bizName.size()));
    forkLocation->set_node_name(std::string(_node.c_str(), _node.size()));
}
}

