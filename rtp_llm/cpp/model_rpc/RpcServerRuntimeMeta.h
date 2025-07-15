#pragma once
#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"

namespace rtp_llm {
class RpcServerRuntimeMeta {
public:
    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finised_version) {
        std::unique_lock<std::mutex> lock(mutex_);
        EngineScheduleInfo           info;
        for (auto& iter : running_streams_) {
            info.running_task_info_list.push_back(iter.second);
        }
        // trimFinishedStreams();
        for (auto& iter : finished_streams_) {
            if (iter.first >= latest_finised_version) {
                info.finished_task_info_list.push_back(iter.second);
            }
        }
        return info;
    }

    void enqueue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        running_streams_[request_id] = EngineScheduleInfo::TaskInfo({request_id,
                                                                     stream->interRequestId(),
                                                                     stream->prefixLength(),
                                                                     stream->inputLength(),
                                                                     stream->getTimeInfo().wait_time_us});
    }

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto                         ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            return;
        }
        auto& task_info         = ptr->second;
        task_info.iterate_count = stream->iterCount();
        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        auto current          = autil::TimeUtility::currentTimeInMilliSeconds();
        task_info.end_time_ms = current;
        finished_streams_.push_back(std::make_pair(current, task_info));
        running_streams_.erase(ptr);
    }

protected:
    void trimFinishedStreams() {
        auto current = autil::TimeUtility::currentTimeInMilliSeconds();
        auto iter    = finished_streams_.begin();
        while (iter != finished_streams_.end()) {
            if (iter->first > current) {
                RTP_LLM_LOG_WARNING("find task: %ld end time: %d bigger than current time: %d",
                                    iter->second.request_id,
                                    iter->first,
                                    current);
                iter = finished_streams_.erase(iter);
            } else if (iter->first + timeout_ms_ <= current) {
                iter = finished_streams_.erase(iter);
            } else {
                break;
            }
        }
    }
    std::unordered_map<int64_t, EngineScheduleInfo::TaskInfo>   running_streams_;
    std::list<std::pair<int64_t, EngineScheduleInfo::TaskInfo>> finished_streams_;
    std::mutex                                                  mutex_;
    int64_t                                                     timeout_ms_        = 5000;
    int64_t                                                     finished_capacity_ = 1000;
};

};  // namespace rtp_llm
