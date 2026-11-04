#pragma once

namespace autil {

class NoCopyable {
protected:
    constexpr NoCopyable() = default;
    NoCopyable(const NoCopyable &) = delete;
    NoCopyable &operator=(const NoCopyable &) = delete;
    NoCopyable(NoCopyable &&) = default;
    NoCopyable &operator=(NoCopyable &&) = default;
};

class NoMoveable {
protected:
    constexpr NoMoveable() = default;
    NoMoveable(const NoMoveable &) = delete;
    NoMoveable &operator=(const NoMoveable &) = delete;
    NoMoveable(NoMoveable &&) = delete;
    NoMoveable &operator=(NoMoveable &&) = delete;
};

} // namespace autil
