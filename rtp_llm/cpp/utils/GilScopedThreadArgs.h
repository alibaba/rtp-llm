#pragma once

#include <atomic>
#include <cassert>
#include <memory>
#include <utility>

#include <pybind11/pybind11.h>

namespace rtp_llm {

// pybind11 2.11 has no thread-local wrapper at all and stores `internals::tstate` as a raw
// TSS key, while pybind11 3.x keeps `thread_specific_storage` under `detail`. Own the storage
// here so the ownership checks do not depend on either internal layout.
class GilThreadStateStorage {
public:
    PyThreadState* get() const noexcept {
        return current();
    }
    void set(PyThreadState* thread_state) noexcept {
        current() = thread_state;
    }
    void reset() noexcept {
        current() = nullptr;
    }

private:
    static PyThreadState*& current() noexcept {
        static thread_local PyThreadState* thread_state = nullptr;
        return thread_state;
    }
};

inline std::atomic<GilThreadStateStorage*>& gilThreadStateStorage() noexcept {
    static std::atomic<GilThreadStateStorage*> storage{nullptr};
    return storage;
}

// This must run while the caller owns the GIL. It records the calling thread's state so
// no-GIL ownership checks stay allocation-free and never enter pybind's slow paths.
//
// get_internals() here is load-bearing, not defensive. pybind needs internals initialized
// under the GIL so internals.tstate exists before any later gil_scoped_acquire, and reaching
// it implicitly (the previous `&get_internals().tstate`) was the only thing doing that. Drop
// it and the process' first get_internals() runs inside a worker's gil_scoped_acquire with
// the GIL released, where it does PyGILState_Ensure/Release and therefore creates *and
// destroys* a PyThreadState while storing it into internals.tstate; gil_scoped_acquire then
// calls PyEval_AcquireThread() on that freed state.
inline void initializeGilThreadStateTracking() {
    pybind11::detail::get_internals();
    static GilThreadStateStorage tracked_storage;
    tracked_storage.set(pybind11::detail::get_thread_state_unchecked());
    gilThreadStateStorage().store(&tracked_storage, std::memory_order_release);
}

inline bool currentThreadHoldsGilFromTrackedState(bool gil_state_check) noexcept {
    if (gil_state_check) {
        return true;
    }

    auto* storage = gilThreadStateStorage().load(std::memory_order_acquire);
    if (storage == nullptr) {
        return false;
    }
    auto* local = storage->get();
    return local != nullptr && local == pybind11::detail::get_thread_state_unchecked();
}

inline bool pythonRuntimeIsFinalizing() noexcept {
#if PY_VERSION_HEX >= 0x03070000 && !defined(PYPY_VERSION)
    return Py_IsInitialized() && _Py_IsFinalizing();
#else
    return false;
#endif
}

inline constexpr bool shouldReleaseGilForBlockingOperation(bool holds_gil, bool runtime_finalizing) noexcept {
    return holds_gil && !runtime_finalizing;
}

inline constexpr bool pythonRuntimeCanAcquireGilFromState(bool initialized, bool runtime_finalizing) noexcept {
    return initialized && !runtime_finalizing;
}

inline bool pythonRuntimeCanAcquireGil() noexcept {
    return pythonRuntimeCanAcquireGilFromState(Py_IsInitialized(), pythonRuntimeIsFinalizing());
}

inline bool currentThreadHoldsGil() noexcept {
    if (!Py_IsInitialized()) {
        return false;
    }
    const bool gil_state_check = PyGILState_Check() != 0;
    if (pythonRuntimeIsFinalizing()) {
        return gil_state_check;
    }
    return currentThreadHoldsGilFromTrackedState(gil_state_check);
}

class GilScopedRelease {
public:
    GilScopedRelease() {
        assert(currentThreadHoldsGil());
        thread_state_ = PyEval_SaveThread();
    }
    ~GilScopedRelease() {
        PyEval_RestoreThread(thread_state_);
    }

    GilScopedRelease(const GilScopedRelease&)            = delete;
    GilScopedRelease& operator=(const GilScopedRelease&) = delete;

private:
    PyThreadState* thread_state_;
};

template<typename Args>
class GilScopedThreadArgs {
public:
    explicit GilScopedThreadArgs(std::shared_ptr<Args> source): gil_(), args_(std::move(*source)) {
        initializeGilThreadStateTracking();
        source.reset();
    }

    GilScopedThreadArgs(const GilScopedThreadArgs&)            = delete;
    GilScopedThreadArgs& operator=(const GilScopedThreadArgs&) = delete;

    Args& get() {
        return args_;
    }

private:
    // Member destruction is reversed, so args_ releases Python references before gil_.
    pybind11::gil_scoped_acquire gil_;
    Args                         args_;
};

}  // namespace rtp_llm
