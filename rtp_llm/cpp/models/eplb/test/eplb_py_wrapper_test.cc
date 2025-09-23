#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"

using namespace rtp_llm;

namespace unittest {

class EplbPyWrapperOP: public torch::jit::CustomClassHolder {
public:
    void init(py::object py_eplb) {
        eplb_py_wrapper_ = ExpertBalancerPythonWrapper(py_eplb);
    }

    void createBalancePlan(torch::Tensor log_stats, torch::Tensor gpu_loads) {
        eplb_py_wrapper_.createBalancePlan(log_stats, gpu_loads, eplb_plan_);
    }

    void loadMoeWeight(int64_t ep_rank, int64_t ep_size) {
        eplb_py_wrapper_.loadBalanceWeight(ep_rank, ep_size, eplb_plan_);
    }

    EplbPlanTensors::tuple_type getResult() {
        return eplb_plan_.to_tuple();
    }

private:
    ExpertBalancerPythonWrapper eplb_py_wrapper_;
    EplbPlanTensors             eplb_plan_;
};

PYBIND11_MODULE(libth_eplb_py_wrapper_test, m) {
    py::class_<EplbPyWrapperOP>(m, "EplbPyWrapperOP")
        .def(py::init<>())
        .def("init", &EplbPyWrapperOP::init)
        .def("create_balance_plan", &EplbPyWrapperOP::createBalancePlan, py::arg("log_stats"), py::arg("gpu_loads"))
        .def("load_moe_weight", &EplbPyWrapperOP::loadMoeWeight, py::arg("ep_rank"), py::arg("ep_size"))
        .def("get_result", &EplbPyWrapperOP::getResult);
}
}  // namespace unittest
