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
    void mtp(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp);
    void runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream);
    void runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp);
private:
    py::object  aiter_module;
    py::object  paged_attention_rocm;
    py::module_ pa_gluon_aot_api;
    py::object  load_all_libs;
    bool use_asm_pa_    = true;
};

void runAiterAsmPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
void runAiterPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
}  // namespace rtp_llm
