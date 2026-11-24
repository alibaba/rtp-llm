#pragma once

#include <ATen/core/Tensor.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace rtp_llm {
namespace {
namespace py = pybind11;

// Tensor object metadata limits from the RTPMM wire codec.
constexpr size_t  kEmbPlanMaxNodes  = 4096;
constexpr int     kEmbPlanMaxDepth  = 32;
constexpr int64_t kEmbPlanAlignment = 64;

inline const char* wireDtype(at::ScalarType type) {
    switch (type) {
        case at::kBool:
            return "torch.bool";
        case at::kByte:
            return "torch.uint8";
        case at::kChar:
            return "torch.int8";
        case at::kShort:
            return "torch.int16";
        case at::kInt:
            return "torch.int32";
        case at::kLong:
            return "torch.int64";
        case at::kHalf:
            return "torch.float16";
        case at::kBFloat16:
            return "torch.bfloat16";
        case at::kFloat:
            return "torch.float32";
        case at::kDouble:
            return "torch.float64";
        default:
            throw py::value_error("unsupported multimodal tensor layout or dtype");
    }
}

class PlanVisitor {
public:
    explicit PlanVisitor(int64_t maxBytes): maxBytes_(maxBytes) {}

    py::object plan(py::handle value) {
        try {
            auto     tree = visit(value, 0);
            py::dict header;
            header["tree"]    = std::move(tree);
            header["tensors"] = specs_;
            return py::make_tuple(std::move(header), std::move(tensors_), size_);
        } catch (const Unsupported&) {
            return py::none();
        }
    }

private:
    struct Unsupported {};

    py::list visit(py::handle value, int depth) {
        if (++nodes_ > kEmbPlanMaxNodes || depth > kEmbPlanMaxDepth) {
            throw py::value_error("multimodal object structure exceeds limits");
        }
        PyObject* object = value.ptr();
        if (THPVariable_CheckExact(object)) {
            auto    prior = identities_.find(object);
            int64_t index;
            if (prior != identities_.end()) {
                index = prior->second;
            } else {
                auto t = py::cast<at::Tensor>(value);
                if (t.layout() != at::kStrided) {
                    throw py::value_error("unsupported multimodal tensor layout or dtype");
                }
                const char* dtype = wireDtype(t.scalar_type());
                if (!t.device().is_cpu() && !t.device().is_cuda()) {
                    throw py::value_error("unsupported multimodal tensor device");
                }
                index                = tensors_.size();
                const int64_t offset = (size_ + kEmbPlanAlignment - 1) / kEmbPlanAlignment * kEmbPlanAlignment;
                if (offset > maxBytes_ || t.numel() > (maxBytes_ - offset) / t.element_size()) {
                    throw py::value_error("multimodal object exceeds byte limit");
                }
                const int64_t nbytes = t.numel() * t.element_size();
                size_                = offset + nbytes;
                identities_.emplace(object, index);
                tensors_.append(value);
                py::dict spec;
                spec["shape"]  = t.sizes().vec();
                spec["dtype"]  = dtype;
                spec["device"] = t.device().is_cuda() ? "cuda" : "cpu";
                spec["offset"] = offset;
                spec["nbytes"] = nbytes;
                specs_.append(std::move(spec));
            }
            return item("tensor", py::int_(index));
        }
        if (PyTuple_CheckExact(object) || PyList_CheckExact(object)) {
            const bool tuple  = PyTuple_CheckExact(object);
            const auto length = tuple ? PyTuple_GET_SIZE(object) : PyList_GET_SIZE(object);
            py::list   values;
            for (Py_ssize_t i = 0; i < length; ++i) {
                auto* child = tuple ? PyTuple_GET_ITEM(object, i) : PyList_GET_ITEM(object, i);
                values.append(visit(py::handle(child), depth + 1));
            }
            return item(tuple ? "tuple" : "list", std::move(values));
        }
        if (PyDict_CheckExact(object)) {
            Py_ssize_t position = 0;
            PyObject*  key;
            PyObject*  child;
            py::list   values;
            while (PyDict_Next(object, &position, &key, &child)) {
                if (!PyUnicode_CheckExact(key)) {
                    throw Unsupported{};
                }
                py::list entry;
                entry.append(py::handle(key));
                entry.append(visit(py::handle(child), depth + 1));
                values.append(std::move(entry));
            }
            return item("dict", std::move(values));
        }
        if (object == Py_None || PyBool_Check(object) || PyLong_CheckExact(object) || PyUnicode_CheckExact(object)
            || (PyFloat_CheckExact(object) && std::isfinite(PyFloat_AS_DOUBLE(object)))) {
            return item("scalar", value);
        }
        throw Unsupported{};
    }

    static py::list item(const char* tag, py::handle value) {
        py::list entry;
        entry.append(tag);
        entry.append(value);
        return entry;
    }

    const int64_t                          maxBytes_;
    int64_t                                size_  = 0;
    size_t                                 nodes_ = 0;
    py::list                               tensors_;
    py::list                               specs_;
    std::unordered_map<PyObject*, int64_t> identities_;
};

}  // namespace

inline void registerEmbTensorObjectPlan(pybind11::module& module) {
    module.def("plan_emb_tensor_object", [](pybind11::handle value, int64_t byteLimit) {
        const auto maxLimit = std::numeric_limits<int64_t>::max() - kEmbPlanAlignment;
        if (byteLimit < 0 || byteLimit > maxLimit) {
            throw pybind11::value_error("invalid multimodal object byte limit");
        }
        return PlanVisitor(byteLimit).plan(value);
    });
}

}  // namespace rtp_llm
