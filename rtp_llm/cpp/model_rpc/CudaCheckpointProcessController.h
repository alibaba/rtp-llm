#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace rtp_llm {

enum class CudaCheckpointProcessState : int {
    RUNNING      = 0,
    LOCKED       = 1,
    CHECKPOINTED = 2,
    FAILED       = 3,
};

class CudaCheckpointDriver {
public:
    virtual ~CudaCheckpointDriver() = default;

    virtual bool ensureLoaded(std::string* error)         = 0;
    virtual int  getState(int pid, int* state)            = 0;
    virtual int  lock(int pid, uint32_t timeout_ms)       = 0;
    virtual int  checkpoint(int pid)                      = 0;
    virtual int  restore(int pid)                         = 0;
    virtual int  unlock(int pid)                          = 0;
};

struct CudaCheckpointCommand {
    std::string action;
    std::string transaction_id;
    int64_t     sleep_epoch{-1};
    uint32_t    lock_timeout_ms{10000};
};

struct CudaCheckpointCommandResult {
    bool        success{false};
    int         cuda_result{-1};
    std::string state{"UNKNOWN"};
    std::string error;
    std::string transaction_id;
    int64_t     sleep_epoch{-1};
};

// Owns the transaction and state-transition rules for checkpointing this
// backend process. The Driver API is injectable so every transition and
// failure path can be tested without requiring a GPU or checkpoint-capable
// NVIDIA driver.
class CudaCheckpointProcessController {
public:
    CudaCheckpointProcessController();
    explicit CudaCheckpointProcessController(std::shared_ptr<CudaCheckpointDriver> driver);

    CudaCheckpointCommandResult execute(const CudaCheckpointCommand& command,
                                        int                          pid,
                                        int64_t                      backend_sleep_epoch,
                                        bool                         backend_sleeping);

    static const char* stateName(int state);

private:
    std::shared_ptr<CudaCheckpointDriver> driver_;
    std::mutex                            mutex_;
    std::string                           transaction_id_;
    int64_t                               sleep_epoch_{-1};
};

}  // namespace rtp_llm
