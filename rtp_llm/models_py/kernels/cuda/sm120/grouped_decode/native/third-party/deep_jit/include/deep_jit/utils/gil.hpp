#pragma once

#include <optional>

#include <pybind11/pybind11.h>

namespace deep_jit {

class GilScopedRelease {
public:
    std::optional<pybind11::gil_scoped_release> release;

    GilScopedRelease() {
        if (Py_IsInitialized() and PyGILState_Check())
            release.emplace();
    }

    GilScopedRelease(const GilScopedRelease&) = delete;
    GilScopedRelease& operator=(const GilScopedRelease&) = delete;
    GilScopedRelease(GilScopedRelease&&) = delete;
    GilScopedRelease& operator=(GilScopedRelease&&) = delete;
};

}  // namespace deep_jit
