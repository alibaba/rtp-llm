#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Type.h"
#include <algorithm>

namespace navi_sched {

template <typename T>
void ClientRequestToMasterData::addVectorSummary(const std::string &tableName,
                                                 const std::string &valueName,
                                                 const std::vector<T> &values,
                                                 size_t limit,
                                                 navi::SummaryBuilder &builder)
{
    auto table = builder.newTable(tableName);
    table->setDesc("total count " + std::to_string(values.size()));
    auto count = std::min(values.size(), limit);
    for (size_t index = 0; index < count; ++index) {
        TABLE_ADD_COLUMN_VALUE(table, "index", index);
        TABLE_ADD_COLUMN_VALUE(table, valueName, values[index]);
    }
}

ClientRequestToMasterData::ClientRequestToMasterData()
    : navi::Data(CLIENT_REQUEST_TO_MASTER_TYPE_ID)
{
}

bool ClientRequestToMasterData::valid() const
{
    return version == NAVI_SCHED_DATA_VERSION && requestId != 0 && !bizName.empty() &&
           timeoutMs > 0 && explicitCacheCheckpointPosition >= 0 &&
           explicitCacheCheckpointPosition <= promptTokenCount &&
           BlockHashUtil::validTokenBlocks(promptTokenCount, blockSize, blockHashes);
}

navi::SummaryErrorCode ClientRequestToMasterData::getSummary(
    const std::string &query, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("ClientRequestToMasterData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "biz_name", bizName);
    TABLE_ADD_COLUMN_VALUE(table, "timeout_ms", timeoutMs);
    TABLE_ADD_COLUMN_VALUE(table, "request_id", requestId);
    TABLE_ADD_COLUMN_VALUE(table, "prompt_token_count", promptTokenCount);
    TABLE_ADD_COLUMN_VALUE(table, "explicit_cache_checkpoint_position",
                           explicitCacheCheckpointPosition);
    TABLE_ADD_COLUMN_VALUE(table, "block_size", blockSize);
    TABLE_ADD_COLUMN_VALUE(table, "block_hash_count", BlockHashUtil::size(blockHashes));
    TABLE_ADD_COLUMN_VALUE(table, "forward_payload_size", forwardPayload.size());
    TABLE_ADD_COLUMN_VALUE(table, "route_only", routeOnly);
    auto level = SummaryUtil::getSummaryLevel(query);
    if (level >= 1) {
        TABLE_ADD_COLUMN_VALUE(table, "collect_replay", collectReplay);
    }
    if (level >= 2) {
        addVectorSummary(
            "ClientBlockHashes", "block_hash", BlockHashUtil::get(blockHashes),
            SummaryUtil::getSummaryLimit(level, BlockHashUtil::size(blockHashes)),
            builder);
    }
    return navi::SEC_OK;
}

void ClientRequestToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(bizName);
    dataBuffer.write(timeoutMs);
    dataBuffer.write(promptTokenCount);
    dataBuffer.write(explicitCacheCheckpointPosition);
    dataBuffer.write(blockSize);
    dataBuffer.write(requestId);
    dataBuffer.write(collectReplay);
    BlockHashUtil::write(dataBuffer, blockHashes);
    dataBuffer.write(forwardPayload);
    dataBuffer.write(routeOnly);
}

void ClientRequestToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    if (version != NAVI_SCHED_DATA_VERSION) {
        return;
    }
    dataBuffer.read(bizName);
    dataBuffer.read(timeoutMs);
    dataBuffer.read(promptTokenCount);
    dataBuffer.read(explicitCacheCheckpointPosition);
    dataBuffer.read(blockSize);
    dataBuffer.read(requestId);
    dataBuffer.read(collectReplay);
    BlockHashUtil::read(dataBuffer, blockHashes);
    dataBuffer.read(forwardPayload);
    dataBuffer.read(routeOnly);
}

class ClientRequestToMasterType : public navi::Type {
public:
    ClientRequestToMasterType()
        : navi::Type(NAVI_TYPE_PARAM, CLIENT_REQUEST_TO_MASTER_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto request = dynamic_cast<ClientRequestToMasterData *>(data.get());
        if (!request) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(request);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        const auto beginTimeUs =
            requestTraceBatchModulus() > 0 ? autil::TimeUtility::currentTime() : 0;
        ClientRequestToMasterData *request = nullptr;
        ctx.getDataBuffer().read(request);
        if (!request) {
            return navi::TEC_NULL_DATA;
        }
        if (request->version != NAVI_SCHED_DATA_VERSION) {
            delete request;
            return navi::TEC_FAILED;
        }
        if (beginTimeUs > 0) {
            request->deserializeBeginTimeUs = beginTimeUs;
            request->deserializeEndTimeUs = autil::TimeUtility::currentTime();
        }
        data.reset(request);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(ClientRequestToMasterType);

}
