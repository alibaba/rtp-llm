#pragma once

#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>

namespace rtp_llm {

// Explicit eager boundaries only. Graph capture/replay uses the model runner's
// capture callbacks. No Python import, GIL acquisition, or tensor clone when off.
inline bool k3TraceEnabled() {
    static const bool enabled = [] {
        const char* root = std::getenv("K3_TRACE_ROOT");
        return root != nullptr && root[0] != '\0';
    }();
    return enabled;
}

inline void k3TraceEvent(const char*                                                  name,
                         std::initializer_list<std::pair<const char*, torch::Tensor>> tensors,
                         std::initializer_list<std::pair<const char*, int64_t>>       details = {}) {
    if (!k3TraceEnabled()) {
        return;
    }
    pybind11::gil_scoped_acquire gil;
    pybind11::dict               values;
    for (const auto& [key, tensor] : tensors) {
        values[key] = tensor.defined() ? pybind11::cast(tensor) : pybind11::none();
    }
    pybind11::dict metadata;
    for (const auto& [key, value] : details) {
        metadata[key] = value;
    }
    pybind11::module_::import("rtp_llm.utils.k3_tensor_trace").attr("event")(name, values, metadata);
}

class K3TraceScope {
public:
    explicit K3TraceScope(const char* name) {
        if (k3TraceEnabled()) {
            pybind11::gil_scoped_acquire gil;
            id_ = pybind11::module_::import("rtp_llm.utils.k3_tensor_trace").attr("enter_scope")(name).cast<int64_t>();
        }
    }
    K3TraceScope(const K3TraceScope&)            = delete;
    K3TraceScope& operator=(const K3TraceScope&) = delete;

    void finish() {
        if (id_ != -1) {
            pybind11::gil_scoped_acquire gil;
            pybind11::module_::import("rtp_llm.utils.k3_tensor_trace").attr("exit_scope")(id_, true);
            id_ = -1;
        }
    }

    ~K3TraceScope() noexcept {
        if (id_ != -1) {
            // Preserve the primary inference/IO exception while marking this
            // scope incomplete. A normal return must call finish() explicitly.
            try {
                pybind11::gil_scoped_acquire gil;
                pybind11::module_::import("rtp_llm.utils.k3_tensor_trace").attr("exit_scope")(id_, false);
            } catch (...) {}
        }
    }

private:
    int64_t id_ = -1;
};

// Snapshot CPU bookkeeping under the stream mutex, then call Python only after
// releasing it. Acquiring the GIL while holding that mutex can deadlock with a
// Python request thread that is waiting to access the same stream.
class K3CpuTraceBuffer {
public:
    void record(const char*                                                  name,
                std::initializer_list<std::pair<const char*, torch::Tensor>> tensors,
                std::initializer_list<std::pair<const char*, int64_t>>       details = {}) {
        if (!k3TraceEnabled()) {
            return;
        }
        Record record{name, {}, details};
        for (const auto& [key, tensor] : tensors) {
            TORCH_CHECK(!tensor.defined() || tensor.device().is_cpu(), "K3CpuTraceBuffer requires CPU tensors");
            record.tensors.emplace_back(key, tensor.defined() ? tensor.detach().clone() : torch::Tensor());
        }
        records_.push_back(std::move(record));
    }

    void flush() {
        if (records_.empty()) {
            return;
        }
        pybind11::gil_scoped_acquire gil;
        auto                         event = pybind11::module_::import("rtp_llm.utils.k3_tensor_trace").attr("event");
        for (const auto& record : records_) {
            pybind11::dict tensors, details;
            for (const auto& [key, tensor] : record.tensors) {
                tensors[key] = tensor.defined() ? pybind11::cast(tensor) : pybind11::none();
            }
            for (const auto& [key, value] : record.details) {
                details[key] = value;
            }
            event(record.name, tensors, details);
        }
        records_.clear();
    }

private:
    struct Record {
        const char*                                        name;
        std::vector<std::pair<const char*, torch::Tensor>> tensors;
        std::vector<std::pair<const char*, int64_t>>       details;
    };
    std::vector<Record> records_;
};

}  // namespace rtp_llm
