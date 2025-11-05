#include <pybind11/pybind11.h>

#include "sampling.h"

namespace py = pybind11;

namespace rtp_llm {

PYBIND11_MODULE(bind, m) {
    m.doc() = "sampling c++ api for test";
    m.def("top_p_renorm_probs", &top_p_renorm_probs, py::arg(), py::arg(), py::arg(), py::arg(), py::arg("stream") = 0, "top_p_renorm_probs");
    m.def("top_k_renorm_probs", &top_k_renorm_probs, py::arg(), py::arg(), py::arg(), py::arg(), py::arg("stream") = 0, "top_k_renorm_probs");
    m.def("top_p_sampling_from_probs", &top_p_sampling_from_probs, py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("stream") = 0, "top_p_sampling_from_probs");
    m.def("top_k_sampling_from_probs", &top_k_sampling_from_probs, py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("stream") = 0, "top_k_sampling_from_probs");
    m.def("top_k_top_p_sampling_from_probs", &top_k_top_p_sampling_from_probs, py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("stream") = 0, "top_k_top_p_sampling_from_probs");
}

}