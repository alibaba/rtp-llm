#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferEngineSelection.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {
namespace {
bool hasDeviceRoute(const GroupSet& group, bool enable_disk_cache) {
    return !group.devicePools().empty() && (group.hostPool() || (enable_disk_cache && group.diskPool()));
}
}  // namespace

std::vector<GroupSetPtr> crcTransferGroups(const std::vector<GroupSetPtr>& groups, bool enable_disk_cache) {
    std::vector<GroupSetPtr> result;
    for (const auto& group : groups)
        if (group->crcEnabled() && hasDeviceRoute(*group, enable_disk_cache))
            result.push_back(group);
    return result;
}

std::string workLoopCompatibility(const std::vector<GroupSetPtr>& groups, bool enable_disk_cache) {
    std::string reason;
    for (size_t index = 0; index < groups.size(); ++index) {
        const auto& group = groups[index];
        if (!group || group->groupSetId() != index || !group->topologyPtr() || group->groupTags().empty()
            || !group->payloadBytes() || group->storageBytes() < group->payloadBytes())
            throw std::invalid_argument("invalid transfer GroupSet mapping or backing layout");
        if (group->crcEnabled()
            && (group->payloadBytes() > std::numeric_limits<size_t>::max() - 19
                || group->storageBytes() != CrcCopyWorkspace::encodedBytes(group->payloadBytes())))
            throw std::invalid_argument("invalid CRC encoded storage geometry");
        if ((group->hostPool()
             && (group->hostPool()->payloadBytes() < group->payloadBytes()
                 || group->hostPool()->strideBytes() < group->storageBytes()))
            || (enable_disk_cache && group->diskPool()
                && (group->diskPool()->payloadBytes() < group->payloadBytes()
                    || group->diskPool()->strideBytes() < group->storageBytes())))
            throw std::invalid_argument("transfer backing cannot hold encoded payload");
        const auto& pools = group->devicePools();
        if (!pools.empty()
            && (pools.size() != group->groupTags().size()
                || std::any_of(pools.begin(), pools.end(), [](const auto& pool) { return !pool; })))
            throw std::invalid_argument("invalid transfer member pools");
        if (!hasDeviceRoute(*group, enable_disk_cache))
            continue;
        if (!group->crcEnabled()) {
            reason = "Device route has no CRC layout";
            continue;
        }
        const int device = pools.front()->deviceIndex();
        if (device < 0 || std::any_of(pools.begin(), pools.end(), [device](const auto& pool) {
                return pool->deviceIndex() != device;
            }))
            throw std::invalid_argument("CRC backings must reside on a single GPU per GroupSet");
        if (!CrcCopyWorkspace::available())
            throw std::invalid_argument("CRC Device routes require the CUDA13 backend");
        // HostBlockPool's allocator guarantees pinned/registered storage.
        (void)getDeviceHostCopyTiles(*group, std::vector<BlockIdxType>(pools.size(), 0));
    }
    return reason;
}

}  // namespace rtp_llm
