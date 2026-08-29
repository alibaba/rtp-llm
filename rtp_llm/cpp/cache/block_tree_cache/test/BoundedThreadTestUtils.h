#pragma once

#include <chrono>
#include <future>
#include <thread>
#include <type_traits>
#include <utility>

namespace rtp_llm::block_tree_cache_test {

// Runs a potentially blocking assertion subject on a dedicated thread. Tests
// join after proving readiness. On a regression timeout, the callable must
// retain shared ownership of every referenced fixture; detaching then keeps
// the test target bounded without destroying live state under the call.
template<typename Result>
class BoundedThread {
public:
    template<typename Function>
    explicit BoundedThread(Function&& function) {
        std::packaged_task<Result()> task(std::forward<Function>(function));
        future_ = task.get_future();
        thread_ = std::thread(std::move(task));
    }

    ~BoundedThread() {
        if (thread_.joinable()) {
            thread_.detach();
        }
    }

    BoundedThread(const BoundedThread&)            = delete;
    BoundedThread& operator=(const BoundedThread&) = delete;

    std::future_status waitFor(std::chrono::milliseconds timeout) {
        return future_.wait_for(timeout);
    }

    Result get() {
        if constexpr (std::is_void_v<Result>) {
            try {
                future_.get();
            } catch (...) {
                thread_.join();
                throw;
            }
            thread_.join();
        } else {
            try {
                Result result = future_.get();
                thread_.join();
                return result;
            } catch (...) {
                thread_.join();
                throw;
            }
        }
    }

private:
    std::future<Result> future_;
    std::thread         thread_;
};

}  // namespace rtp_llm::block_tree_cache_test
