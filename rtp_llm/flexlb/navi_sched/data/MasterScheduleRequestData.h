#pragma once

#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Data.h"

namespace navi_sched {

class MasterScheduleRequestData : public navi::Data {
public:
    MasterScheduleRequestData(const ClientRequestToMasterDataConstPtr &request,
                              const navi::AsyncPipePtr &responsePipe);
    bool valid() const;
    bool pending() const;
    bool finish(const navi::DataPtr &response);
    void cancel();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    ClientRequestToMasterDataConstPtr request;
    int64_t enqueueTimeUs = 0;

private:
    class ResponseState;
    NAVI_TYPEDEF_PTR(ResponseState);

private:
    ResponseStatePtr _responseState;
};

NAVI_TYPEDEF_PTR(MasterScheduleRequestData);

}
