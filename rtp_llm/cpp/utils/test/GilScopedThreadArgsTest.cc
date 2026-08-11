#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/utils/GilScopedThreadArgs.h"

namespace rtp_llm {
namespace {

struct PythonRefProbe {
    PythonRefProbe(pybind11::object value, std::shared_ptr<std::promise<bool>> destroyed_with_gil):
        value(std::move(value)), destroyed_with_gil(std::move(destroyed_with_gil)) {}

    PythonRefProbe(PythonRefProbe&&) noexcept            = default;
    PythonRefProbe& operator=(PythonRefProbe&&) noexcept = default;
    PythonRefProbe(const PythonRefProbe&)                = delete;
    PythonRefProbe& operator=(const PythonRefProbe&)     = delete;

    ~PythonRefProbe() {
        if (destroyed_with_gil) {
            auto* thread_state = PyThreadState_Get();
            auto* marker       = PyLong_FromLong(1);
            const bool python_api_succeeded = marker != nullptr;
            Py_XDECREF(marker);
            destroyed_with_gil->set_value(currentThreadHoldsGil() && thread_state != nullptr && python_api_succeeded);
        }
    }

    pybind11::object                   value;
    std::shared_ptr<std::promise<bool>> destroyed_with_gil;
};

TEST(GilScopedThreadArgsTest, FinalizingRuntimeDoesNotTransitionGilOwnership) {
    EXPECT_TRUE(shouldReleaseGilForBlockingOperation(true, false));
    EXPECT_FALSE(shouldReleaseGilForBlockingOperation(false, false));
    EXPECT_FALSE(shouldReleaseGilForBlockingOperation(true, true));

    EXPECT_TRUE(pythonRuntimeCanAcquireGilFromState(true, false));
    EXPECT_FALSE(pythonRuntimeCanAcquireGilFromState(false, false));
    EXPECT_FALSE(pythonRuntimeCanAcquireGilFromState(true, true));
}

TEST(GilScopedThreadArgsTest, DestroysPythonReferencesWhileHoldingGil) {
    pybind11::scoped_interpreter interpreter;
    initializeGilThreadStateTracking();
    ASSERT_TRUE(currentThreadHoldsGil());

    std::promise<void> worker_started;
    auto               worker_started_result = worker_started.get_future();
    std::promise<void> check_ownership;
    auto               check_ownership_result = check_ownership.get_future();
    std::promise<bool> worker_holds_gil;
    auto               worker_holds_gil_result = worker_holds_gil.get_future();
    std::thread raw_worker([&]() {
        worker_started.set_value();
        check_ownership_result.wait();
        worker_holds_gil.set_value(currentThreadHoldsGil());
    });

    EXPECT_EQ(worker_started_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    check_ownership.set_value();
    EXPECT_EQ(worker_holds_gil_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    raw_worker.join();
    EXPECT_FALSE(worker_holds_gil_result.get());
    EXPECT_TRUE(currentThreadHoldsGil());

    auto* interpreter_state = PyThreadState_Get()->interp;
    std::thread custom_gil_worker([interpreter_state]() {
        auto* thread_state = PyThreadState_New(interpreter_state);
        ASSERT_NE(thread_state, nullptr);
        auto* storage = gilThreadStateStorage().load(std::memory_order_acquire);
        ASSERT_NE(storage, nullptr);
        storage->set(thread_state);
        PyEval_AcquireThread(thread_state);

        EXPECT_TRUE(currentThreadHoldsGilFromTrackedState(false));
        EXPECT_TRUE(currentThreadHoldsGil());
        {
            GilScopedRelease release;
            EXPECT_FALSE(currentThreadHoldsGilFromTrackedState(false));
            EXPECT_FALSE(currentThreadHoldsGil());
        }
        EXPECT_TRUE(currentThreadHoldsGil());

        PyThreadState_Clear(thread_state);
        storage->reset();
        PyThreadState_DeleteCurrent();
    });
    {
        GilScopedRelease release;
        custom_gil_worker.join();
    }
    EXPECT_TRUE(currentThreadHoldsGil());

    auto force_destruction_signal = std::make_shared<std::promise<bool>>();
    auto force_destruction_result = force_destruction_signal->get_future();
    auto force_owned_probe = std::make_unique<PythonRefProbe>(pybind11::none(), force_destruction_signal);
    auto* force_probe      = force_owned_probe.release();
    {
        GilScopedRelease release;
        pybind11::gil_scoped_acquire acquire;
        std::unique_ptr<PythonRefProbe> owner(force_probe);
    }
    EXPECT_TRUE(force_destruction_result.get());

    auto destruction_signal = std::make_shared<std::promise<bool>>();
    auto destruction_result = destruction_signal->get_future();
    auto args = std::make_shared<PythonRefProbe>(pybind11::none(), destruction_signal);

    std::thread worker([args]() mutable {
        GilScopedThreadArgs<PythonRefProbe> scoped_args(std::move(args));
        EXPECT_TRUE(currentThreadHoldsGil());
        EXPECT_NE(PyThreadState_Get(), nullptr);
        EXPECT_TRUE(scoped_args.get().value.is_none());
        {
            GilScopedRelease release;
            EXPECT_FALSE(currentThreadHoldsGil());
        }
        EXPECT_TRUE(currentThreadHoldsGil());
    });

    {
        GilScopedRelease release;
        EXPECT_FALSE(currentThreadHoldsGil());
        EXPECT_EQ(destruction_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        worker.join();
    }

    EXPECT_TRUE(destruction_result.get());
}

}  // namespace
}  // namespace rtp_llm
