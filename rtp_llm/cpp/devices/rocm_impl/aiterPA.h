#pragma once
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
// #include "aiter_meta/csrc/include/attention.h"
#include "attention.h"
#include "attention_asm.h"
#include <pybind11/pybind11.h>
#include <Python.h>

namespace py = pybind11;

namespace rtp_llm {

class AiterWrapper {
public:
    AiterWrapper(const DeviceInitParams& params);
    void runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream);
    void runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp, hipStream_t stream);
private:
    py::module_ pa_gluon_aot_api;
    py::module_ hip_pa_api;
    py::object  pa_gluon_load_libs;
    py::object  hip_pa_load_libs;
};

void runAiterAsmPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
void runAiterPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
}  // namespace rtp_llm
