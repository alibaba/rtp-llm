#include "aios/navi_sched/data/MasterScheduleRequestData.h"
#include "autil/TimeUtility.h"
#include <atomic>

namespace navi_sched {
const std::string MASTER_SCHEDULE_REQUEST_TYPE_ID = "NaviSchedMasterScheduleRequestType";

class MasterScheduleRequestData::ResponseState {
public:
    explicit ResponseState(const navi::AsyncPipePtr &responsePipe)
        : _responsePipe(responsePipe)
    {
    }

    ~ResponseState() { cancel(); }

public:
    bool valid() const { return bool(_responsePipe); }
    bool pending() const { return !_finished.load(std::memory_order_acquire); }

    bool finish(const navi::DataPtr &response)
    {
        if (!_responsePipe || !response) {
            return false;
        }
        bool expected = false;
        if (!_finished.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
            return false;
        }
        auto ec = _responsePipe->setData(response);
        _responsePipe->setEof();
        return navi::EC_NONE == ec;
    }

    void cancel()
    {
        bool expected = false;
        if (_responsePipe && _finished.compare_exchange_strong(
                                 expected, true, std::memory_order_acq_rel)) {
            _responsePipe->setEof();
        }
    }

private:
    navi::AsyncPipePtr _responsePipe;
    std::atomic<bool> _finished{false};
};

MasterScheduleRequestData::MasterScheduleRequestData(
    const ClientRequestToMasterDataConstPtr &requestData, const navi::AsyncPipePtr &pipe)
    : navi::Data(MASTER_SCHEDULE_REQUEST_TYPE_ID)
    , request(requestData)
    , enqueueTimeUs(autil::TimeUtility::monotonicTimeUs())
    , _responseState(std::make_shared<ResponseState>(pipe))
{
}

bool MasterScheduleRequestData::valid() const
{
    return request && _responseState && _responseState->valid();
}

bool MasterScheduleRequestData::finish(const navi::DataPtr &response)
{
    return _responseState && response && _responseState->finish(response);
}

bool MasterScheduleRequestData::pending() const
{
    return valid() && _responseState->pending();
}

void MasterScheduleRequestData::cancel()
{
    if (_responseState) {
        _responseState->cancel();
    }
}

navi::SummaryErrorCode MasterScheduleRequestData::getSummary(
    const std::string &query, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterScheduleRequestData");
    TABLE_ADD_COLUMN_VALUE(table, "schedule_request_id",
                           request ? request->requestId : 0);
    TABLE_ADD_COLUMN_VALUE(table, "request_id", request ? request->requestId : 0);
    TABLE_ADD_COLUMN_VALUE(table, "biz_name", request ? request->bizName : "");
    TABLE_ADD_COLUMN_VALUE(table, "enqueue_time_us", enqueueTimeUs);
    TABLE_ADD_COLUMN_VALUE(table, "has_response_pipe",
                           _responseState && _responseState->valid());
    return navi::SEC_OK;
}

}
