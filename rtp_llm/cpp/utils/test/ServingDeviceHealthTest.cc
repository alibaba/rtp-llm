#include "rtp_llm/cpp/utils/ServingDeviceHealth.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {
template<typename T>
struct FailingDiagnosticAllocator {
    using value_type             = T;
    FailingDiagnosticAllocator() = default;
    template<typename U>
    FailingDiagnosticAllocator(const FailingDiagnosticAllocator<U>&) {}
    T* allocate(std::size_t) {
        throw std::bad_alloc();
    }
    void deallocate(T*, std::size_t) noexcept {}
};

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

// The production boundary invokes these in place of native GetDevice and an
// event/stream/fence API. The stale host-thread last-error belongs to device A,
// while current/resource/serving may all be healthy device B.
struct InjectedRuntime {
    int current{1};
    int get_code{0};
    int operation_code{0};
    int stale_last_error{700};
    int operation_calls{0};
    int peek_calls{0};
    int getDevice(int& device) {
        device = current;
        return get_code;
    }
    int invoke() {
        ++operation_calls;
        return operation_code;
    }
    int peek() {
        ++peek_calls;
        return stale_last_error;
    }
};

rtp_llm::ServingRuntimeResult run(rtp_llm::ServingDeviceHealth&   health,
                                  InjectedRuntime&                runtime,
                                  rtp_llm::DeviceRuntimeDomain    domain,
                                  rtp_llm::DeviceRuntimeOperation operation,
                                  int                             resource = 1,
                                  int                             serving  = 1) {
    return rtp_llm::runServingDeviceOperation(
        &health,
        domain,
        serving,
        resource,
        operation,
        [&](int& device) { return runtime.getDevice(device); },
        [&] { return runtime.invoke(); });
}
}  // namespace

int main() {
    using namespace rtp_llm;
    static_assert(static_cast<int>(ErrorCode::SERVICE_UNHEALTHY) == 8215);
    static_assert(static_cast<int>(ErrorCode::EXECUTION_EXCEPTION) == 606);
    for (auto domain : {DeviceRuntimeDomain::CUDA, DeviceRuntimeDomain::HIP}) {
        for (int code : {0, 2, 9, 17, 600, 701, 702, 999, -200, -1}) {
            ServingDeviceHealth health;
            InjectedRuntime     runtime;
            runtime.operation_code = code;
            const auto result      = run(health, runtime, domain, DeviceRuntimeOperation::STREAM_QUERY);
            require(result.operation_performed && result.fault.code == code, "lost native result");
            require(health.healthy() && !health.firstFault(), "stale foreign last-error poisoned serving device");
            require(runtime.peek_calls == 0 && runtime.operation_calls == 1, "boundary touched thread last-error");
            require(runtime.stale_last_error == 700, "boundary cleared foreign last-error");
        }
        for (auto operation : {DeviceRuntimeOperation::GET_DEVICE,
                               DeviceRuntimeOperation::SET_DEVICE,
                               DeviceRuntimeOperation::RESTORE_DEVICE,
                               DeviceRuntimeOperation::PEEK_LAST_ERROR}) {
            ServingDeviceHealth health;
            InjectedRuntime     runtime;
            runtime.operation_code = 700;
            const auto result      = run(health, runtime, domain, operation);
            require(result.fault.code == 700 && result.fault.operation == operation,
                    "unattributed native code lost diagnostics");
            require(health.healthy(), "selection/restore/peek was treated as serving-context evidence");
        }
        {
            ServingDeviceHealth health;
            InjectedRuntime     runtime;
            runtime.get_code  = 700;
            const auto result = run(health, runtime, domain, DeviceRuntimeOperation::STREAM_WAIT);
            require(!result.operation_performed && result.fault.operation == DeviceRuntimeOperation::GET_DEVICE
                        && result.fault.device == -1 && result.fault.code == 700,
                    "failed GetDevice was relabeled as a bound stream failure");
            require(health.healthy() && runtime.operation_calls == 0, "unknown device poisoned service");
        }
        for (int current : {0, 1}) {
            ServingDeviceHealth health;
            InjectedRuntime     runtime;
            runtime.current        = current;
            runtime.operation_code = 700;
            const auto result      = run(health, runtime, domain, DeviceRuntimeOperation::STREAM_WAIT, 0, 1);
            require(health.healthy(), "foreign resource poisoned serving context");
            require(result.operation_performed == (current == 0), "executed with an unverified handle device");
        }
        for (auto operation : {DeviceRuntimeOperation::EVENT_WAIT,
                               DeviceRuntimeOperation::STREAM_WAIT,
                               DeviceRuntimeOperation::STREAM_QUERY,
                               DeviceRuntimeOperation::DEVICE_WAIT}) {
            for (int code : {700, 710, 719}) {
                ServingDeviceHealth health;
                InjectedRuntime     runtime;
                runtime.operation_code = code;
                const auto result      = run(health, runtime, domain, operation);
                require(result.operation_performed && !health.healthy(), "bound sticky failure not latched");
                // Force a real diagnostic-string allocation failure after the
                // production boundary returns; do not exhaust process memory.
                bool allocation_failed = false;
                try {
                    std::basic_string<char, std::char_traits<char>, FailingDiagnosticAllocator<char>> diagnostic(4096,
                                                                                                                 'x');
                } catch (const std::bad_alloc&) {
                    allocation_failed = true;
                }
                require(allocation_failed, "diagnostic allocator did not fail");
                const auto first = health.firstFault();
                require(!health.healthy() && first && first->domain == domain && first->code == code
                            && first->device == 1 && first->operation == operation,
                        "diagnostic allocation failure lost sticky typed evidence");
                runtime.operation_code = 0;
                run(health, runtime, domain, operation);
                require(!health.healthy() && health.firstFault()->code == code, "success cleared first fault");
            }
        }
    }
    {
        ServingDeviceHealth health;
        InjectedRuntime     runtime;
        runtime.operation_code = 700;
        run(health, runtime, DeviceRuntimeDomain::UNKNOWN, DeviceRuntimeOperation::EVENT_WAIT);
        require(health.healthy(), "unknown runtime domain accepted");
        require(!health.observeRuntimeResult({DeviceRuntimeDomain::CUDA, 700, -1, DeviceRuntimeOperation::EVENT_WAIT}),
                "unknown device accepted");
    }
    auto            publisher = std::make_shared<ServingDeviceHealth>();
    InjectedRuntime runtime;
    runtime.operation_code = 710;
    run(*publisher, runtime, DeviceRuntimeDomain::CUDA, DeviceRuntimeOperation::EVENT_WAIT);
    auto engine = publisher;
    publisher.reset();
    require(!engine->healthy() && engine->firstFault()->device == 1, "cached health lost startup fault");

    ServingDeviceHealth      concurrent;
    std::atomic<bool>        start{false};
    std::atomic<int>         finished{0};
    std::vector<std::thread> writers;
    for (int i = 0; i < 8; ++i) {
        writers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            InjectedRuntime runtime;
            runtime.operation_code = 700;
            run(concurrent, runtime, DeviceRuntimeDomain::HIP, DeviceRuntimeOperation::STREAM_WAIT);
            finished.fetch_add(1, std::memory_order_release);
        });
    }
    start.store(true, std::memory_order_release);
    bool observed_failure = false;
    while (finished.load(std::memory_order_acquire) != 8) {
        const bool healthy = concurrent.healthy();
        require(!observed_failure || !healthy, "sticky health regressed");
        observed_failure |= !healthy;
        const auto fault = concurrent.firstFault();
        if (fault) {
            require(fault->domain == DeviceRuntimeDomain::HIP && fault->code == 700 && fault->device == 1
                        && fault->operation == DeviceRuntimeOperation::STREAM_WAIT,
                    "torn metadata");
        }
    }
    for (auto& writer : writers) {
        writer.join();
    }
    require(!concurrent.healthy() && concurrent.firstFault().has_value(), "concurrent latch missing");
    require(ErrorCodeToString(ErrorCode::SERVICE_UNHEALTHY) == "SERVICE_UNHEALTHY", "missing error name");
    std::cout << "ServingDeviceHealth attributed runtime-boundary tests passed\n";
}
