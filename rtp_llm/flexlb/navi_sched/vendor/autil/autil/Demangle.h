#pragma once

#include <cxxabi.h>
#include <memory>
#include <string>

namespace autil {

inline std::string demangle(const char *name) {
    int status = -4;
    std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
    return (status == 0) ? res.get() : name;
}

template <typename T>
inline std::string demangle() {
    return demangle(typeid(T).name());
}

} // namespace autil
