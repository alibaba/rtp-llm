#include "rtp_llm/cpp/distribute/EngineLoopCoordinator.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace rtp_llm {
namespace {
constexpr uint64_t kMagic    = 0x52545053544f5031;
constexpr int      kMaxRanks = 256;
}  // namespace

struct EngineLoopCoordinator::SharedState {
    uint64_t magic;
    int      size;
    bool     complete;
    bool     released;
    struct Rank {
        int      pid;
        uint64_t started;
        uint64_t completed;
        bool     parked;
        bool     drained;
        bool     stop_ready;
    } ranks[kMaxRanks];
};

class EngineLoopCoordinator::Guard {
public:
    explicit Guard(EngineLoopCoordinator& owner): owner_(owner), lock_(owner.mutex_) {
        const auto deadline = std::chrono::steady_clock::now() + owner.timeout_;
        while (flock(owner_.fd_, LOCK_EX | LOCK_NB) < 0) {
            if (errno != EWOULDBLOCK && errno != EINTR) {
                throw std::runtime_error("engine shutdown state lock failed");
            }
            owner_.checkDeadline(deadline, "state lock");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ~Guard() {
        flock(owner_.fd_, LOCK_UN);
    }

private:
    EngineLoopCoordinator&       owner_;
    std::unique_lock<std::mutex> lock_;
};

EngineLoopCoordinator::EngineLoopCoordinator(std::string path, int rank, int size, std::chrono::milliseconds timeout):
    path_(std::move(path)), rank_(rank), size_(size), timeout_(timeout) {
    if (size < 2 || size > kMaxRanks || rank < 0 || rank >= size || timeout.count() <= 0) {
        throw std::invalid_argument("invalid local engine shutdown geometry");
    }
    fd_ = open(path_.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("cannot open engine shutdown state");
    }
    try {
        Guard       guard(*this);
        struct stat info {};
        if (fstat(fd_, &info) || !S_ISREG(info.st_mode) || info.st_uid != geteuid() || (info.st_mode & 0077)
            || info.st_nlink != 1) {
            throw std::runtime_error("unsafe engine shutdown state file");
        }
        const bool fresh = info.st_size == 0;
        if ((fresh && ftruncate(fd_, sizeof(SharedState)))
            || (!fresh && info.st_size != static_cast<off_t>(sizeof(SharedState)))) {
            throw std::runtime_error("engine shutdown state layout differs");
        }
        void* address = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (address == MAP_FAILED) {
            throw std::runtime_error("cannot map engine shutdown state");
        }
        state_ = static_cast<SharedState*>(address);
        if (fresh) {
            std::memset(state_, 0, sizeof(*state_));
            state_->magic = kMagic;
            state_->size  = size;
        }
        if (state_->magic != kMagic || state_->size != size || state_->complete || state_->ranks[rank].pid != 0) {
            throw std::runtime_error("engine shutdown state belongs to another engine epoch");
        }
        state_->ranks[rank].pid = getpid();
    } catch (...) {
        if (state_) {
            munmap(state_, sizeof(*state_));
            state_ = nullptr;
        }
        close(fd_);
        fd_ = -1;
        throw;
    }
}

EngineLoopCoordinator::~EngineLoopCoordinator() {
    if (state_) {
        munmap(state_, sizeof(*state_));
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

std::unique_ptr<EngineLoopCoordinator>
EngineLoopCoordinator::createFromEnvironment(int rank, int size, int local_size) {
    const char* base = std::getenv("RTP_LLM_ENGINE_LOOP_COORDINATOR_BASE_PATH");
    if (size <= 1 || size != local_size || base == nullptr || *base == '\0') {
        return nullptr;
    }
    static std::mutex                      epochs_mutex;
    static std::map<std::string, uint64_t> epochs;
    std::lock_guard<std::mutex>            lock(epochs_mutex);
    std::string path = std::string(base) + ".engine" + std::to_string(++epochs[base]) + ".state";
    return std::make_unique<EngineLoopCoordinator>(path, rank, size);
}

bool EngineLoopCoordinator::supportsLocalGroup(int world_size, int local_size, int tp_size, int ep_size) {
    return world_size > 1 && world_size == local_size && tp_size > 1 && ep_size == world_size;
}

void EngineLoopCoordinator::requestStop() {
    Guard guard(*this);
    auto& self   = state_->ranks[rank_];
    self.started = std::max(self.started, started_.load());
}

void EngineLoopCoordinator::waitForStopReady(std::chrono::milliseconds drain_timeout) {
    const auto deadline = std::chrono::steady_clock::now() + drain_timeout + timeout_;
    while (true) {
        {
            Guard guard(*this);
            state_->ranks[rank_].stop_ready = true;
            bool all_ready                  = true;
            for (int rank = 0; rank < size_; ++rank) {
                all_ready = all_ready && state_->ranks[rank].stop_ready;
            }
            if (all_ready) {
                return;
            }
        }
        checkDeadline(deadline, "ingress drain");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void EngineLoopCoordinator::checkDeadline(const std::chrono::steady_clock::time_point& deadline,
                                          const char*                                  phase) const {
    if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(std::string("engine shutdown rendezvous timed out: ") + phase);
    }
}

bool EngineLoopCoordinator::beforeStep(const std::atomic<bool>& running) {
    if (!stopping_) {
        // Publish before reading running: stop() must see a step that was
        // admitted before its stop latch, even if that step is still blocked.
        started_.store(completed_ + 1);
        if (running.load()) {
            return true;
        }
        stopping_ = true;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
        {
            Guard guard(*this);
            auto& self          = state_->ranks[rank_];
            self.started        = std::max(self.started, started_.load());
            self.completed      = completed_;
            self.parked         = true;
            uint64_t target     = 0;
            bool     all_parked = true;
            for (int rank = 0; rank < size_; ++rank) {
                const auto& peer = state_->ranks[rank];
                target           = std::max(target, peer.started);
                all_parked       = all_parked && peer.pid != 0 && peer.parked;
            }
            bool same_step = all_parked;
            for (int rank = 0; rank < size_; ++rank) {
                same_step = same_step && state_->ranks[rank].completed == target;
            }
            if (same_step) {
                state_->complete = true;
            }
            if (state_->complete) {
                return false;
            }
            if (completed_ < target) {
                self.parked = false;
                started_.store(completed_ + 1);
                self.started = completed_ + 1;
                return true;
            }
        }
        checkDeadline(deadline, "model steps");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void EngineLoopCoordinator::afterStep() {
    ++completed_;
}

void EngineLoopCoordinator::waitForGpuDrain() {
    if (gpu_drained_) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
        {
            Guard guard(*this);
            if (!state_->complete) {
                throw std::logic_error("GPU drain preceded common engine loop exit");
            }
            state_->ranks[rank_].drained = true;
            bool all_drained             = true;
            for (int rank = 0; rank < size_; ++rank) {
                all_drained = all_drained && state_->ranks[rank].drained;
            }
            if (all_drained) {
                if (!state_->released) {
                    if (unlink(path_.c_str()) != 0 && errno != ENOENT) {
                        throw std::runtime_error("cannot retire engine shutdown state");
                    }
                    state_->released = true;
                }
                gpu_drained_ = true;
                return;
            }
        }
        checkDeadline(deadline, "GPU drain");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace rtp_llm
