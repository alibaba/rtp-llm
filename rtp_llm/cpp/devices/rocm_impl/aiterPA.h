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
    void set_triton_pa(bool v) { use_triton_pa_ = v; }
private:
    py::object aiter_module;
    py::object paged_attention_rocm;
    py::object aiter_triton_module;
    py::object pa_decode_gluon;
    py::object triton_tl;
    py::object compute_type_;
    bool use_asm_pa_    = true;
    bool use_triton_pa_ = true;
};

void runAiterAsmPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
void runAiterPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
}  // namespace rtp_llm
