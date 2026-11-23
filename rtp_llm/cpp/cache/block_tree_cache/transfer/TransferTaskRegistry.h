#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace rtp_llm {

class TransferTaskRegistry {
public:
    enum class State { UNKNOWN, RUNNING, BUFFER_SAFE, REJECTED };
    enum class ForgetResult { FAILED, DONE, RUNNING, INVALID };
    using QueryCallback = std::function<void(std::chrono::steady_clock::time_point)>;
    using ForgetCallback = std::function<void(bool)>;

    virtual ~TransferTaskRegistry() = default;
    virtual bool registerOnWorkerTaskWithTraceID(const std::string& id, QueryCallback offeredQuery = {},
                                                  ForgetCallback offeredForget = {}) = 0;
    virtual bool ioFinishedOnWorkerWithTraceID(const std::string& id) = 0;
    virtual std::vector<State> queryIfIoFinishedInOneWorkerWithTraceIDs(const std::vector<std::string>& ids) = 0;
    virtual std::vector<ForgetResult> forgetTaskInfoOnWorker(const std::vector<std::string>& ids, bool forced = false) = 0;
};

}
