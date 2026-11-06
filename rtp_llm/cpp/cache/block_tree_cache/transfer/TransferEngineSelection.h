#pragma once

#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// Validate common geometry first. An empty reason means the entire engine can
// use WorkLoop. Resource limits are checked only by the selected implementation.
std::string              workLoopCompatibility(const std::vector<GroupSetPtr>& groups, bool enable_disk_cache);
std::vector<GroupSetPtr> crcTransferGroups(const std::vector<GroupSetPtr>& groups, bool enable_disk_cache);

}  // namespace rtp_llm
