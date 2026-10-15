#pragma once

#include <cstddef>
#include <functional>

namespace rtp_llm {

// Stable identity of one cache-owning module. Id 0 is the main model; ids
// 1..n identify draft/MTP slots. The value is intentionally not implicit: all
// call sites must state whether they are passing a module identity or another
// integer such as an engine model id or a layer index.
struct CacheModuleId {
public:
    explicit constexpr CacheModuleId(size_t value = 0): value_(value) {}

    static constexpr CacheModuleId main() {
        return CacheModuleId(0);
    }

    constexpr bool isMain() const {
        return value_ == 0;
    }

    constexpr size_t value() const {
        return value_;
    }

    friend constexpr bool operator==(const CacheModuleId& lhs, const CacheModuleId& rhs) {
        return lhs.value_ == rhs.value_;
    }

    friend constexpr bool operator!=(const CacheModuleId& lhs, const CacheModuleId& rhs) {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(const CacheModuleId& lhs, const CacheModuleId& rhs) {
        return lhs.value_ < rhs.value_;
    }

private:
    size_t value_;
};

}  // namespace rtp_llm

namespace std {
template<>
struct hash<rtp_llm::CacheModuleId> {
    size_t operator()(const rtp_llm::CacheModuleId& id) const noexcept {
        return hash<size_t>()(id.value());
    }
};
}  // namespace std
