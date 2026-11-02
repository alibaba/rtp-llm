#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace navi_sched {

static const std::string CLIENT_REQUEST_TO_MASTER_TYPE_ID = "NaviSchedClientType";
static const std::string CLIENT_REQUEST_TO_MASTER_DATA_NAME =
    "navi_sched.client_request_to_master_data";
class ClientRequestToMasterData
    : public navi::Data,
      public std::enable_shared_from_this<ClientRequestToMasterData> {
public:
    ClientRequestToMasterData();
    bool valid() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

private:
    template <typename T>
    static void addVectorSummary(const std::string &tableName,
                                 const std::string &valueName,
                                 const std::vector<T> &values, size_t limit,
                                 navi::SummaryBuilder &builder);

public:
    int32_t version = NAVI_SCHED_DATA_VERSION;
    std::string bizName;
    int64_t timeoutMs = 0;
    int64_t promptTokenCount = 0;
    int64_t blockSize = 256;
    uint64_t requestId = 0;
    bool collectReplay = false;
    NaviSchedBlockHashesConstPtr blockHashes;
    std::string forwardPayload;
    int64_t deserializeBeginTimeUs = 0;
    int64_t deserializeEndTimeUs = 0;
};

using ClientRequestToMasterDataConstPtr =
    std::shared_ptr<const ClientRequestToMasterData>;

}
