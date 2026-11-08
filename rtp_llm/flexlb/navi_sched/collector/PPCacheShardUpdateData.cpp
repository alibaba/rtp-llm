#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "navi/engine/Type.h"
#include <limits>
#include <stdexcept>

namespace navi_sched {

const std::string PP_CACHE_SHARD_UPDATE_TYPE_ID = "NaviSchedPPCacheShardUpdateType";

PPCacheShardUpdateData::PPCacheShardUpdateData()
    : navi::Data(PP_CACHE_SHARD_UPDATE_TYPE_ID)
{
}

bool PPCacheShardUpdateData::valid() const
{
    if (version != VERSION || !update.streamGeneration || !update.sequence ||
        update.peerCacheReadAvailable < -1 || update.peerCacheReadAvailable > 1 ||
        update.terminals.size() > std::numeric_limits<uint32_t>::max() ||
        (update.fullSnapshot &&
         (!update.removedFull.empty() || !update.removedTail.empty())))
        return false;
    for (const auto &terminal : update.terminals) {
        if (!terminal.request.requestId || !terminal.request.attemptId ||
            terminal.status < PPShardRequestStatus::SUCCEEDED ||
            terminal.status > PPShardRequestStatus::CANCELLED)
            return false;
    }
    return true;
}

void PPCacheShardUpdateData::serialize(autil::DataBuffer &buffer) const
{
    if (!valid()) throw std::invalid_argument("invalid PP cache shard update");
    buffer.write(version);
    buffer.write(update.shardId);
    buffer.write(update.streamGeneration);
    buffer.write(update.sequence);
    buffer.write(update.fullSnapshot);
    buffer.write(update.peerCacheReadAvailable);
    buffer.write(update.addedFull);
    buffer.write(update.removedFull);
    buffer.write(update.addedTail);
    buffer.write(update.removedTail);
    buffer.write(static_cast<uint32_t>(update.terminals.size()));
    for (const auto &terminal : update.terminals) {
        buffer.write(terminal.request.requestId);
        buffer.write(terminal.request.attemptId);
        buffer.write(static_cast<uint8_t>(terminal.status));
    }
}

void PPCacheShardUpdateData::deserialize(autil::DataBuffer &buffer)
{
    update = {};
    buffer.read(version);
    if (version != VERSION) return;
    buffer.read(update.shardId);
    buffer.read(update.streamGeneration);
    buffer.read(update.sequence);
    buffer.read(update.fullSnapshot);
    buffer.read(update.peerCacheReadAvailable);
    buffer.read(update.addedFull);
    buffer.read(update.removedFull);
    buffer.read(update.addedTail);
    buffer.read(update.removedTail);
    uint32_t count = 0;
    buffer.read(count);
    update.terminals.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        PPShardRequestTerminal terminal;
        uint8_t status = 0;
        buffer.read(terminal.request.requestId);
        buffer.read(terminal.request.attemptId);
        buffer.read(status);
        terminal.status = static_cast<PPShardRequestStatus>(status);
        update.terminals.push_back(terminal);
    }
}

navi::SummaryErrorCode PPCacheShardUpdateData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("PPCacheShardUpdateData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "shard_id", update.shardId);
    TABLE_ADD_COLUMN_VALUE(table, "stream_generation", update.streamGeneration);
    TABLE_ADD_COLUMN_VALUE(table, "sequence", update.sequence);
    TABLE_ADD_COLUMN_VALUE(table, "full_snapshot", update.fullSnapshot);
    TABLE_ADD_COLUMN_VALUE(table, "peer_cache_read_available",
                           int(update.peerCacheReadAvailable));
    TABLE_ADD_COLUMN_VALUE(table, "terminal_count", update.terminals.size());
    TABLE_ADD_COLUMN_VALUE(table, "added_full_count", update.addedFull.size());
    TABLE_ADD_COLUMN_VALUE(table, "removed_full_count", update.removedFull.size());
    TABLE_ADD_COLUMN_VALUE(table, "added_tail_count", update.addedTail.size());
    TABLE_ADD_COLUMN_VALUE(table, "removed_tail_count", update.removedTail.size());
    return navi::SEC_OK;
}

class PPCacheShardUpdateType : public navi::Type {
public:
    PPCacheShardUpdateType() : navi::Type(NAVI_TYPE_PARAM, PP_CACHE_SHARD_UPDATE_TYPE_ID)
    {
    }
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto update = dynamic_cast<PPCacheShardUpdateData *>(data.get());
        if (!update || !update->valid()) return navi::TEC_TYPE_MISMATCH;
        ctx.getDataBuffer().write(update);
        return navi::TEC_NONE;
    }
    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        PPCacheShardUpdateData *update = nullptr;
        ctx.getDataBuffer().read(update);
        if (!update) return navi::TEC_NULL_DATA;
        data.reset(update);
        return update->valid() ? navi::TEC_NONE : navi::TEC_TYPE_MISMATCH;
    }
};

REGISTER_TYPE(PPCacheShardUpdateType);

}
