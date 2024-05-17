#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace rtp_llm {

EngineBase::EngineBase(const MagaInitParams& params) {
    initDevices(params);
}

void EngineBase::initDevices(const MagaInitParams& params) {
    auto global_params = ft::DeviceFactory::getDefaultGlobalDeviceParams();
    auto& default_device_params = global_params.device_params[0].second;
    default_device_params.tp_size = params.gpt_init_parameter->tp_size_;
    default_device_params.tp_rank = params.gpt_init_parameter->tp_rank_;
    default_device_params.master_ip = params.gpt_init_parameter->nccl_ip_;
    default_device_params.master_port = params.gpt_init_parameter->nccl_port_;
    default_device_params.max_batch_size = params.gpt_init_parameter->max_context_batch_size_
                                         + params.gpt_init_parameter->max_generate_batch_size_;
    ft::DeviceFactory::initDevices(global_params);
    device_ = ft::DeviceFactory::getDefaultDevice();
}

}  // namespace rtp_llm
