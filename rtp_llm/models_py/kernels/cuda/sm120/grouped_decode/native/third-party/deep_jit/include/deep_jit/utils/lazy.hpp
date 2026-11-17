#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <dlfcn.h>

#include <deep_jit/utils/exception.hpp>

#define DJ_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace deep_jit {

template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;

public:
    explicit LazyInit(std::nullptr_t) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    T* operator->() {
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr)
            ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        return ptr.get();
    }

    std::shared_ptr<T> get() {
        (void)operator->();
        return ptr;
    }
};

}  // namespace deep_jit

#define DJ_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                         \
    inline void* handle_func_name() {                                                         \
        static void* handle = [] {                                                            \
            ::dlerror();                                                                      \
            void* value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);                                \
            if (value == nullptr) {                                                           \
                const char* error = ::dlerror();                                              \
                DJ_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error); \
            }                                                                                 \
            return value;                                                                     \
        }();                                                                                  \
        return handle;                                                                        \
    }

#define DJ_STRINGIFY(name) #name

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                                                   \
    template <typename... Args>                                                                                            \
    static auto lazy_##name(Args&&... args) {                                                                              \
        static const auto func = []() {                                                                                    \
            void* symbol = ::dlsym(handle_func_name(), DJ_STRINGIFY(name));                                                \
            if (symbol == nullptr) {                                                                                       \
                const char* error = ::dlerror();                                                                           \
                DJ_PANIC("failed to load {} from {}: {}", DJ_STRINGIFY(name),                                              \
                         #handle_func_name, error == nullptr ? "unknown" : error);                                         \
            }                                                                                                              \
            return reinterpret_cast<decltype(&name)>(symbol);                                                              \
        }();                                                                                                               \
        return func(std::forward<Args>(args)...);                                                                          \
    }
