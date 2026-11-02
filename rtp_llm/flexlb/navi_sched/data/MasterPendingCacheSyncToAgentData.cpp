#include "aios/navi_sched/data/MasterPendingCacheSyncToAgentData.h"

#include "navi/engine/Type.h"

namespace navi_sched {

MasterPendingCacheSyncToAgentData::MasterPendingCacheSyncToAgentData()
    : navi::Data(MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID)
{
}

navi::SummaryErrorCode MasterPendingCacheSyncToAgentData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterPendingCacheSyncToAgentData");
    TABLE_ADD_COLUMN_VALUE(table, "collector_id", collectorId);
    TABLE_ADD_COLUMN_VALUE(table, "reason", reason);
    TABLE_ADD_COLUMN_VALUE(table, "full_snapshot", fullSnapshot);
    TABLE_ADD_COLUMN_VALUE(table, "snapshot_ready_before", snapshotReadyBefore);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_state_computed",
                           candidateStateComputed);
    TABLE_ADD_COLUMN_VALUE(table, "master_revision_before",
                           masterRevisionBefore);
    TABLE_ADD_COLUMN_VALUE(table, "expected_revision", expectedRevision);
    TABLE_ADD_COLUMN_VALUE(table, "rejected_revision", rejectedRevision);
    TABLE_ADD_COLUMN_VALUE(table, "master_checksum_before",
                           masterChecksumBefore);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_checksum", candidateChecksum);
    TABLE_ADD_COLUMN_VALUE(table, "received_checksum", receivedChecksum);
    TABLE_ADD_COLUMN_VALUE(table, "master_batch_count_before",
                           masterBatchCountBefore);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_batch_count", candidateBatchCount);
    TABLE_ADD_COLUMN_VALUE(table, "received_batch_count", receivedBatchCount);
    TABLE_ADD_COLUMN_VALUE(table, "master_block_count_before",
                           masterBlockCountBefore);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_block_count", candidateBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "received_block_count", receivedBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "master_tail_block_count_before",
                           masterTailBlockCountBefore);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_tail_block_count",
                           candidateTailBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "received_tail_block_count",
                           receivedTailBlockCount);
    return navi::SEC_OK;
}

void MasterPendingCacheSyncToAgentData::serialize(
    autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(collectorId);
    dataBuffer.write(reason);
    dataBuffer.write(fullSnapshot);
    dataBuffer.write(snapshotReadyBefore);
    dataBuffer.write(candidateStateComputed);
    dataBuffer.write(masterRevisionBefore);
    dataBuffer.write(expectedRevision);
    dataBuffer.write(rejectedRevision);
    dataBuffer.write(masterChecksumBefore);
    dataBuffer.write(candidateChecksum);
    dataBuffer.write(receivedChecksum);
    dataBuffer.write(masterBatchCountBefore);
    dataBuffer.write(candidateBatchCount);
    dataBuffer.write(receivedBatchCount);
    dataBuffer.write(masterBlockCountBefore);
    dataBuffer.write(candidateBlockCount);
    dataBuffer.write(receivedBlockCount);
    dataBuffer.write(masterTailBlockCountBefore);
    dataBuffer.write(candidateTailBlockCount);
    dataBuffer.write(receivedTailBlockCount);
}

void MasterPendingCacheSyncToAgentData::deserialize(
    autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(collectorId);
    dataBuffer.read(reason);
    dataBuffer.read(fullSnapshot);
    dataBuffer.read(snapshotReadyBefore);
    dataBuffer.read(candidateStateComputed);
    dataBuffer.read(masterRevisionBefore);
    dataBuffer.read(expectedRevision);
    dataBuffer.read(rejectedRevision);
    dataBuffer.read(masterChecksumBefore);
    dataBuffer.read(candidateChecksum);
    dataBuffer.read(receivedChecksum);
    dataBuffer.read(masterBatchCountBefore);
    dataBuffer.read(candidateBatchCount);
    dataBuffer.read(receivedBatchCount);
    dataBuffer.read(masterBlockCountBefore);
    dataBuffer.read(candidateBlockCount);
    dataBuffer.read(receivedBlockCount);
    dataBuffer.read(masterTailBlockCountBefore);
    dataBuffer.read(candidateTailBlockCount);
    dataBuffer.read(receivedTailBlockCount);
}

class MasterPendingCacheSyncToAgentType : public navi::Type {
public:
    MasterPendingCacheSyncToAgentType()
        : navi::Type(NAVI_TYPE_PARAM, MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto request =
            dynamic_cast<MasterPendingCacheSyncToAgentData *>(data.get());
        if (!request) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(request);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        MasterPendingCacheSyncToAgentData *request = nullptr;
        ctx.getDataBuffer().read(request);
        if (!request) {
            return navi::TEC_NULL_DATA;
        }
        data.reset(request);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(MasterPendingCacheSyncToAgentType);

}
