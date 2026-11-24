#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteCopyFailureTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include <cuda_runtime.h>
#include <new>
namespace rtp_llm::block_transfer_engine_test {
namespace {
std::mutex                         armed_mutex;
std::shared_ptr<RemoteCopyFailure> armed;
std::atomic<bool>                  registration_armed{false};
std::shared_ptr<RemoteCopyFailure> current() {
    std::lock_guard<std::mutex> lock(armed_mutex);
    return armed;
}
}  // namespace
void RemoteCopyFailure::open() {
    std::lock_guard<std::mutex> lock(mutex);
    opened = true;
    cv.notify_all();
}
bool RemoteCopyFailure::waitQueued() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return queued; });
}
bool RemoteCopyFailure::waitInjected() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return injected; });
}
std::shared_ptr<RemoteCopyFailure> armRemoteCopyFailure(int registration, bool second_copy) {
    auto state               = std::make_shared<RemoteCopyFailure>();
    state->fail_registration = registration;
    state->fail_second_copy  = second_copy;
    std::lock_guard<std::mutex> lock(armed_mutex);
    armed              = state;
    registration_armed = registration != 0;
    return state;
}
extern "C" cudaError_t __real_cudaFreeHost(void*);
void                   retireRemoteCopyFailure(const std::shared_ptr<RemoteCopyFailure>& state) {
    state->open();
    cudaDeviceSynchronize();
    {
        std::lock_guard<std::mutex> lock(armed_mutex);
        armed.reset();
        registration_armed = false;
    }
    for (auto* ptr : state->deferred_frees)
        __real_cudaFreeHost(ptr);
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* ptr) {
    auto state = current();
    if (state) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->queued && !state->opened) {
            ++state->early_frees;
            state->deferred_frees.push_back(ptr);
            return cudaSuccess;
        }
    }
    return __real_cudaFreeHost(ptr);
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t
__wrap_cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    auto state = current();
    if (state) {
        std::unique_lock<std::mutex> lock(state->mutex);
        const auto                   copy = ++state->copies;
        if (copy == 1) {
            auto* keep  = new std::shared_ptr<RemoteCopyFailure>(state);
            auto  error = cudaLaunchHostFunc(
                stream,
                [](void* arg) {
                    std::unique_ptr<std::shared_ptr<RemoteCopyFailure>> keep(
                        static_cast<std::shared_ptr<RemoteCopyFailure>*>(arg));
                    auto                         state = *keep;
                    std::unique_lock<std::mutex> lock(state->mutex);
                    state->queued = true;
                    state->cv.notify_all();
                    state->cv.wait_for(lock, std::chrono::seconds(10), [&] { return state->opened; });
                },
                keep);
            if (error != cudaSuccess) {
                delete keep;
                return error;
            }
        }
        if (copy == 2 && state->fail_second_copy) {
            state->injected = true;
            state->cv.notify_all();
            return cudaErrorInvalidValue;
        }
    }
    return __real_cudaMemcpyAsync(dst, src, bytes, kind, stream);
}
bool remoteCopyRegistrationArmed() {
    return registration_armed.load();
}
void remoteCopyRegistrationAllocation() {
    auto state = current();
    if (state && state->fail_registration) {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (++state->registrations == state->fail_registration) {
            state->cv.wait_for(lock, std::chrono::milliseconds(300), [&] { return state->queued; });
            state->injected = true;
            state->cv.notify_all();
            throw std::bad_alloc();
        }
    }
}
}  // namespace rtp_llm::block_transfer_engine_test
