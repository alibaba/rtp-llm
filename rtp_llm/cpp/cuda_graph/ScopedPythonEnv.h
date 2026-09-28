#pragma once

#include <cstdlib>
#include <exception>
#include <string>

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

class ScopedPythonEnvFlag {
public:
    ScopedPythonEnvFlag(const char* name, const char* value): name_(name) {
        pybind11::gil_scoped_acquire gil;
        if (const char* old_process_value = std::getenv(name_.c_str())) {
            had_process_value_ = true;
            old_process_value_ = old_process_value;
        }
        auto environ   = pybind11::module_::import("os").attr("environ");
        auto old_value = environ.attr("get")(name_, pybind11::none());
        had_old_value_ = !old_value.is_none();
        if (had_old_value_) {
            old_value_ = old_value.cast<std::string>();
        }
        // Python readers use this mapping; libc setenv alone leaves it stale.
        environ[pybind11::str(name_)] = pybind11::str(value);
    }

    ~ScopedPythonEnvFlag() noexcept {
        try {
            pybind11::gil_scoped_acquire gil;
            auto                         environ = pybind11::module_::import("os").attr("environ");
            if (had_old_value_) {
                environ[pybind11::str(name_)] = pybind11::str(old_value_);
            } else {
                environ.attr("pop")(name_, pybind11::none());
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("Failed to restore Python environment flag %s: %s", name_.c_str(), error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("Failed to restore Python environment flag %s: unknown error", name_.c_str());
        }
        // Preserve a pre-existing libc/Python mismatch as well as the mapping.
        const int result =
            had_process_value_ ? setenv(name_.c_str(), old_process_value_.c_str(), 1) : unsetenv(name_.c_str());
        if (result != 0) {
            RTP_LLM_LOG_ERROR("Failed to restore process environment flag %s", name_.c_str());
        }
    }

    ScopedPythonEnvFlag(const ScopedPythonEnvFlag&)            = delete;
    ScopedPythonEnvFlag& operator=(const ScopedPythonEnvFlag&) = delete;

private:
    std::string name_;
    bool        had_old_value_ = false;
    std::string old_value_;
    bool        had_process_value_ = false;
    std::string old_process_value_;
};

}  // namespace rtp_llm
