#include "rtp_llm/cpp/model_rpc/CudaCheckpointProcessController.h"

#include <array>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <utility>

namespace rtp_llm {
namespace {

constexpr int kCudaSuccess  = 0;
constexpr int kRunning      = static_cast<int>(CudaCheckpointProcessState::RUNNING);
constexpr int kLocked       = static_cast<int>(CudaCheckpointProcessState::LOCKED);
constexpr int kCheckpointed = static_cast<int>(CudaCheckpointProcessState::CHECKPOINTED);

#if USING_CUDA
// Keep this binding dynamic. RTP-LLM still builds against CUDA toolkits whose
// cuda.h predates the process-checkpoint declarations, while the capability is
// supplied by the installed NVIDIA driver (libcuda.so.1).
class DynamicCudaCheckpointDriver final : public CudaCheckpointDriver {
public:
    using InitFn      = int (*)(unsigned int);
    using GetStateFn  = int (*)(int, int*);
    using OperationFn = int (*)(int, void*);

    bool ensureLoaded(std::string* error) override {
        std::call_once(load_once_, [this]() {
            library_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
            if (library_ == nullptr) {
                const char* loader_error = dlerror();
                load_error_ = std::string("dlopen(libcuda.so.1) failed: ")
                              + (loader_error == nullptr ? "unknown loader error" : loader_error);
                return;
            }
            init_       = resolve<InitFn>("cuInit");
            get_state_  = resolve<GetStateFn>("cuCheckpointProcessGetState");
            lock_       = resolve<OperationFn>("cuCheckpointProcessLock");
            checkpoint_ = resolve<OperationFn>("cuCheckpointProcessCheckpoint");
            restore_    = resolve<OperationFn>("cuCheckpointProcessRestore");
            unlock_     = resolve<OperationFn>("cuCheckpointProcessUnlock");
            if (!init_ || !get_state_ || !lock_ || !checkpoint_ || !restore_ || !unlock_) {
                load_error_ = "libcuda.so.1 does not export the complete CUDA checkpoint API";
                return;
            }
            const int result = init_(0);
            if (result != kCudaSuccess) {
                load_error_ = "cuInit failed with CUresult=" + std::to_string(result);
                return;
            }
            loaded_ = true;
        });
        if (!loaded_ && error != nullptr) {
            *error = load_error_;
        }
        return loaded_;
    }

    int getState(int pid, int* state) override {
        return get_state_(pid, state);
    }

    int lock(int pid, uint32_t timeout_ms) override {
        alignas(uint64_t) std::array<unsigned char, 64> args{};
        std::memcpy(args.data(), &timeout_ms, sizeof(timeout_ms));
        return lock_(pid, args.data());
    }

    int checkpoint(int pid) override {
        alignas(uint64_t) std::array<unsigned char, 64> args{};
        return checkpoint_(pid, args.data());
    }

    int restore(int pid) override {
        alignas(uint64_t) std::array<unsigned char, 64> args{};
        return restore_(pid, args.data());
    }

    int unlock(int pid) override {
        alignas(uint64_t) std::array<unsigned char, 64> args{};
        return unlock_(pid, args.data());
    }

private:
    template<typename T>
    T resolve(const char* symbol) {
        return reinterpret_cast<T>(dlsym(library_, symbol));
    }

    std::once_flag load_once_;
    void*          library_{nullptr};
    InitFn         init_{nullptr};
    GetStateFn     get_state_{nullptr};
    OperationFn    lock_{nullptr};
    OperationFn    checkpoint_{nullptr};
    OperationFn    restore_{nullptr};
    OperationFn    unlock_{nullptr};
    bool           loaded_{false};
    std::string    load_error_;
};
#else
class DynamicCudaCheckpointDriver final : public CudaCheckpointDriver {
public:
    bool ensureLoaded(std::string* error) override {
        if (error != nullptr) {
            *error = "CUDA process checkpoint is unavailable on a non-CUDA build";
        }
        return false;
    }
    int getState(int, int*) override {
        return -1;
    }
    int lock(int, uint32_t) override {
        return -1;
    }
    int checkpoint(int) override {
        return -1;
    }
    int restore(int) override {
        return -1;
    }
    int unlock(int) override {
        return -1;
    }
};
#endif

std::shared_ptr<CudaCheckpointDriver> makeCudaCheckpointDriver() {
    return std::make_shared<DynamicCudaCheckpointDriver>();
}

bool isMutatingAction(const std::string& action) {
    return action == "LOCK" || action == "CHECKPOINT" || action == "RESTORE" || action == "UNLOCK";
}

}  // namespace

CudaCheckpointProcessController::CudaCheckpointProcessController():
    CudaCheckpointProcessController(makeCudaCheckpointDriver()) {}

CudaCheckpointProcessController::CudaCheckpointProcessController(std::shared_ptr<CudaCheckpointDriver> driver):
    driver_(std::move(driver)) {}

const char* CudaCheckpointProcessController::stateName(int state) {
    switch (state) {
        case static_cast<int>(CudaCheckpointProcessState::RUNNING):
            return "RUNNING";
        case static_cast<int>(CudaCheckpointProcessState::LOCKED):
            return "LOCKED";
        case static_cast<int>(CudaCheckpointProcessState::CHECKPOINTED):
            return "CHECKPOINTED";
        case static_cast<int>(CudaCheckpointProcessState::FAILED):
            return "FAILED";
        default:
            return "UNKNOWN";
    }
}

CudaCheckpointCommandResult CudaCheckpointProcessController::execute(const CudaCheckpointCommand& command,
                                                                     int                          pid,
                                                                     int64_t                      backend_sleep_epoch,
                                                                     bool                         backend_sleeping) {
    std::lock_guard<std::mutex> guard(mutex_);
    CudaCheckpointCommandResult response;

    const auto set_transaction = [&]() {
        response.transaction_id = transaction_id_;
        response.sleep_epoch    = sleep_epoch_;
    };
    set_transaction();

    std::string load_error;
    if (!driver_ || !driver_->ensureLoaded(&load_error)) {
        response.error = load_error.empty() ? "CUDA checkpoint driver is unavailable" : load_error;
        return response;
    }

    int state  = -1;
    int result = driver_->getState(pid, &state);
    response.cuda_result = result;
    response.state       = stateName(state);
    if (result != kCudaSuccess) {
        response.error = "cuCheckpointProcessGetState failed with CUresult=" + std::to_string(result);
        return response;
    }

    if (command.action == "GET_STATE") {
        response.success = true;
        return response;
    }
    if (!isMutatingAction(command.action)) {
        response.error = "invalid CUDA checkpoint action: " + command.action;
        return response;
    }
    if (command.transaction_id.empty() || command.sleep_epoch < 0) {
        response.error = "mutating CUDA checkpoint requests require transaction_id and non-negative sleep_epoch";
        return response;
    }
    if (!backend_sleeping) {
        response.error = "CUDA checkpoint mutation requires backend state SLEEPING";
        return response;
    }
    if (command.sleep_epoch != backend_sleep_epoch) {
        response.error = "CUDA checkpoint sleep_epoch differs from backend sleep state";
        return response;
    }

    const bool same_transaction =
        transaction_id_ == command.transaction_id && sleep_epoch_ == command.sleep_epoch;
    if (command.action == "LOCK" && state == kRunning) {
        // A RUNNING process may start a new transaction. Persist the identity
        // before Lock so it is part of CPU state throughout the operation.
        transaction_id_ = command.transaction_id;
        sleep_epoch_     = command.sleep_epoch;
        set_transaction();
    } else if (!same_transaction) {
        response.error = "CUDA checkpoint transaction does not own this backend rank";
        return response;
    }

    bool already_complete = false;
    if (command.action == "LOCK") {
        already_complete = state == kLocked;
        if (!already_complete && state == kRunning) {
            result = driver_->lock(pid, command.lock_timeout_ms == 0 ? 10000 : command.lock_timeout_ms);
        } else if (!already_complete) {
            response.error = "LOCK requires CUDA process state RUNNING or LOCKED";
            return response;
        }
    } else if (command.action == "CHECKPOINT") {
        already_complete = state == kCheckpointed;
        if (!already_complete && state == kLocked) {
            result = driver_->checkpoint(pid);
        } else if (!already_complete) {
            response.error = "CHECKPOINT requires CUDA process state LOCKED or CHECKPOINTED";
            return response;
        }
    } else if (command.action == "RESTORE") {
        already_complete = state == kLocked;
        if (!already_complete && state == kCheckpointed) {
            result = driver_->restore(pid);
        } else if (!already_complete) {
            response.error = "RESTORE requires CUDA process state CHECKPOINTED or LOCKED";
            return response;
        }
    } else {
        already_complete = state == kRunning;
        if (!already_complete && state == kLocked) {
            result = driver_->unlock(pid);
        } else if (!already_complete) {
            response.error = "UNLOCK requires CUDA process state LOCKED or RUNNING";
            return response;
        }
    }

    if (!already_complete && result != kCudaSuccess) {
        response.cuda_result = result;
        response.error       = command.action + " failed with CUresult=" + std::to_string(result);
        return response;
    }

    // GetState is part of the checkpoint API and remains legal while normal
    // CUDA entry points are blocked. Avoid cuGetErrorString/Name until UNLOCK.
    result               = driver_->getState(pid, &state);
    response.cuda_result = result;
    response.state       = stateName(state);
    set_transaction();
    if (result != kCudaSuccess) {
        response.error = "post-action cuCheckpointProcessGetState failed with CUresult=" + std::to_string(result);
        return response;
    }
    const int expected_state = command.action == "CHECKPOINT" ? kCheckpointed
                                : command.action == "UNLOCK"   ? kRunning
                                                               : kLocked;
    if (state != expected_state) {
        response.error = command.action + " returned without reaching the expected CUDA process state";
        return response;
    }
    response.success = true;
    return response;
}

}  // namespace rtp_llm
