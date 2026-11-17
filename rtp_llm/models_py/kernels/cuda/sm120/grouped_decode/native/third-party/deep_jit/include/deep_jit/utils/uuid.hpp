#pragma once

#include <chrono>
#include <fmt/format.h>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>

namespace deep_jit {

// Process-unique id: pid + three random words. The generator is thread_local
// and seeded per thread so concurrent callers never collide.
inline std::string get_uuid() {
    thread_local std::mt19937 generator([]() {
        std::random_device rd;
        return static_cast<std::mt19937::result_type>(rd()) ^
            static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<std::mt19937::result_type>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }());
    thread_local std::uniform_int_distribution<uint32_t> distribution;
    return fmt::format("{}-{:08x}-{:08x}-{:08x}", ::getpid(), distribution(generator), distribution(generator), distribution(generator));
}

}  // namespace deep_jit
