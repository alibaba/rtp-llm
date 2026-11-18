#include "aios/navi_sched/data/V6dCacheTouchData.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "navi/engine/Type.h"

namespace navi_sched {

void V6dCacheTouchData::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(version);
    buffer.write(requestId);
    buffer.write(blockHashes);
}

void V6dCacheTouchData::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(version);
    if (version != NAVI_SCHED_DATA_VERSION) {
        return;
    }
    buffer.read(requestId);
    buffer.read(blockHashes);
}

navi::SummaryErrorCode V6dCacheTouchData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("V6dCacheTouchData");
    TABLE_ADD_COLUMN_VALUE(table, "request_id", requestId);
    TABLE_ADD_COLUMN_VALUE(table, "block_count", blockHashes.size());
    return navi::SEC_OK;
}

class V6dCacheTouchType : public navi::Type {
public:
    V6dCacheTouchType()
        : navi::Type(NAVI_TYPE_PARAM, V6D_CACHE_TOUCH_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto response = dynamic_cast<V6dCacheTouchData *>(data.get());
        if (!response) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(response);
        return navi::TEC_NONE;
    }
    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        V6dCacheTouchData *response = nullptr;
        ctx.getDataBuffer().read(response);
        if (!response) {
            return navi::TEC_NULL_DATA;
        }
        if (response->version != NAVI_SCHED_DATA_VERSION) {
            delete response;
            return navi::TEC_FAILED;
        }
        data.reset(response);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(V6dCacheTouchType);


}
