#pragma once

#include <atomic>

namespace rtp_llm {

class AtomicGuard {
public:
    AtomicGuard(std::atomic<size_t>& atomic_var): atomic_var_(atomic_var) {
        atomic_var_++;
    }

    ~AtomicGuard() {
        atomic_var_--;
    }

private:
    std::atomic<size_t>& atomic_var_;
};

typedef std::shared_ptr<AtomicGuard> AtomicGuardPtr;

}  // namespace rtp_llm
