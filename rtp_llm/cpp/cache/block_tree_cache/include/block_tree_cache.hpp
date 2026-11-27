#ifndef BLOCK_TREE_CACHE_HPP
#define BLOCK_TREE_CACHE_HPP

#include "block_tree_cache.h"

#include <memory>
#include <string_view>
#include <type_traits>

namespace block_tree_cache {

inline btc_bytes bytes(std::string_view value) {
    return {value.data(), value.size()};
}

template<class T, void (*free)(T*)>
using Result = std::unique_ptr<T, std::integral_constant<decltype(free), free>>;

using Plan = Result<btc_plan, btc_plan_free>;
using Batch = Result<btc_batch, btc_batch_free>;

// The engine closes its cache explicitly after acknowledging all operations.
class CacheRef {
public:
    explicit CacheRef(btc_cache* handle): handle_(handle) {}

    btc_cache* get() const { return handle_; }

    btc_status match(uint64_t request, btc_keys keys, const btc_limits& limits, Plan& result) const {
        btc_plan* plan = nullptr;
        const auto status = btc_match(handle_, request, keys, &limits, &plan);
        result.reset(plan);
        return status;
    }

    btc_status pollMatch(uint64_t request, Plan& result) const {
        btc_plan* plan = nullptr;
        const auto status = btc_poll_match(handle_, request, &plan);
        result.reset(plan);
        return status;
    }

    btc_status bind(Plan& plan, const btc_binding* bindings, size_t count) const {
        return btc_bind(handle_, plan.get(), bindings, count);
    }

    btc_status poll(Batch& result) const {
        btc_batch* batch = nullptr;
        const auto status = btc_poll(handle_, &batch);
        result.reset(batch);
        return status;
    }

private:
    btc_cache* handle_;
};

}  // namespace block_tree_cache

#endif
