#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

CrcDumpWriter::Metadata CrcDumpWriter::dumpMetadata(const TransferDescriptor& descriptor, const GroupSet& group) {
    const bool              store = descriptor.source_tier == Tier::DEVICE;
    CrcDumpWriter::Metadata metadata{
        {"device", std::to_string(group.devicePools().front()->deviceIndex())},
        {"operation", store ? "store" : "load"},
        {"direction", std::string(tierName(descriptor.source_tier)) + "->" + tierName(descriptor.target_tier)},
        {"copy_item", descriptor.debugString()},
        {"group_set_id", std::to_string(descriptor.group_set_id)},
        {"kind", cacheGroupTypeName(group.groupType())},
        {"layout", group.usesPhysicalPayloadGeometry() ? "physical_pool_geometry" : "topology_geometry"},
        {"member_count", std::to_string(group.groupTags().size())},
        // CRC diagnostics are emitted after gather/store or load/scatter has completed.
        {"output_written", "true"},
        {"cpu_snapshot_role", store ? "candidate_output_after_failure" : "restore_source_after_failure"},
        {"gpu_footer_role", store ? "candidate_output" : "checked_source"},
        {"gpu_snapshot_role", store ? "gathered_candidate_output" : "checked_source"},
        {"tile_copy_stage", store ? "gather_completed" : "scatter_completed"},
        {"cpu_backing",
         descriptor.source_tier == Tier::HOST || descriptor.target_tier == Tier::HOST ? "memory_pool" :
                                                                                        "disk_staging_buffer"},
    };
    for (size_t member = 0; member < group.groupTags().size(); ++member) {
        const auto& tag                                = group.groupTags()[member];
        const auto& base                               = group.group(tag);
        const auto  prefix                             = "member_" + std::to_string(member) + "_";
        metadata[prefix + "tag"]                       = tag;
        metadata[prefix + "kind"]                      = cacheGroupTypeName(base.policy.group_type);
        metadata[prefix + "pool"]                      = group.devicePools()[member]->poolName();
        metadata[prefix + "seq_size_per_block"]        = std::to_string(base.seqSizePerBlock());
        metadata[prefix + "kernel_seq_size_per_block"] = std::to_string(base.kernelSeqSizePerBlock());
        metadata[prefix + "kv_stride_bytes"]           = std::to_string(base.kvBlockStrideBytes());
        metadata[prefix + "scale_stride_bytes"]        = std::to_string(base.kvScaleStrideBytes());
    }
    // HOST<->DISK validates host pool records without a device endpoint. Block zero gives
    // the same immutable geometry; its addresses are intentionally not recorded.
    const bool has_device = descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE;
    const auto blocks =
        has_device ? descriptor.blocksAt(Tier::DEVICE) : std::vector<BlockIdxType>(group.devicePools().size(), 0);
    const auto tiles = getDeviceHostCopyTiles(group, blocks);
    for (size_t index = 0; index < tiles.size(); ++index) {
        const auto& tile            = tiles[index];
        const auto  prefix          = "layout_tile_" + std::to_string(index) + "_";
        const auto& tag             = group.groupTags()[tile.member_group_id];
        metadata[prefix + "member"] = std::to_string(tile.member_group_id);
        metadata[prefix + "tag"]    = tag;
        metadata[prefix + "layer_id"] =
            std::to_string(group.topologyPtr()->layerIdsForGroup(tag)[tile.local_layer_index]);
        metadata[prefix + "offset"] = std::to_string(tile.host_offset);
        metadata[prefix + "bytes"]  = std::to_string(tile.buffer.size_bytes);
    }
    metadata["layout_tile_count"] = std::to_string(tiles.size());
    for (size_t i = 0; i < descriptor.source_blocks.size(); ++i)
        metadata["source_block_" + std::to_string(i)] = std::to_string(descriptor.source_blocks[i]);
    for (size_t i = 0; i < descriptor.target_blocks.size(); ++i)
        metadata["target_block_" + std::to_string(i)] = std::to_string(descriptor.target_blocks[i]);
    if (group.hostPool())
        metadata["host_pool_stride_bytes"] = std::to_string(group.hostPool()->strideBytes());
    if (group.diskPool()) {
        metadata["disk_file"]              = group.diskPool()->filePath();
        metadata["disk_pool_stride_bytes"] = std::to_string(group.diskPool()->strideBytes());
        if (descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK) {
            const auto block        = descriptor.singleBlockAt(Tier::DISK);
            metadata["disk_slot"]   = std::to_string(block);
            metadata["disk_offset"] = std::to_string(group.diskPool()->blockOffset(block));
        }
    }
    return metadata;
}

}  // namespace rtp_llm
