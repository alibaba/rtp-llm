#pragma once
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/EngineScheduleInfo.h"

namespace rtp_llm {
class PrefillRpcServerRuntimeMeta {
public:
    EngineScheduleInfo getEngineScheduleInfo() {
        std::unique_lock<std::mutex> lock(mutex_);
        EngineScheduleInfo           info;
        for (auto& iter : running_streams_) {
            info.running_task_info_list.push_back(iter.second);
        }
        trimFinishedStreams();
        for (auto& iter : finished_streams_) {
            info.finished_task_info_list.push_back(iter.second);
        }
        return info;
    }

    void enqueue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        running_streams_[request_id] =
            EngineScheduleInfo::TaskInfo({request_id, stream->prefixLength(), stream->inputLength()});
    }

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto                         ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            return;
        }
        auto& task_info = ptr->second;
        finished_streams_.push_back(std::make_pair(autil::TimeUtility::currentTimeInMilliSeconds(), task_info));
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
    int64_t                                                     timeout_ms_ = 5000;
};

};  // namespace rtp_llm
