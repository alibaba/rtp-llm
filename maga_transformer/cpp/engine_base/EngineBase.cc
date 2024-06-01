#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    EngineBase::initDevices(params);
    device_ = ft::DeviceFactory::getDefaultDevice();
}

void EngineBase::initDevices(const EngineInitParams& params) {
    auto global_params = ft::DeviceFactory::getDefaultGlobalDeviceParams();
    auto& default_device_params = global_params.device_params[0].second;
    default_device_params.tp_size = params.gpt_init_parameter.tp_size_;
    default_device_params.tp_rank = params.gpt_init_parameter.tp_rank_;
    default_device_params.master_ip = params.gpt_init_parameter.nccl_ip_;
    default_device_params.master_port = params.gpt_init_parameter.nccl_port_;
    default_device_params.max_batch_size = params.gpt_init_parameter.max_context_batch_size_
                                         + params.gpt_init_parameter.max_generate_batch_size_;
    default_device_params.device_reserve_memory_bytes = -128L * 1024 * 1024; // 64MB
    default_device_params.host_reserve_memory_bytes = 2L * 1024 * 1024 * 1024; // 2GB
    if (params.gpt_init_parameter.reserve_runtime_mem_mb_) {
        default_device_params.device_reserve_memory_bytes = -params.gpt_init_parameter.reserve_runtime_mem_mb_ * 1024 * 1024L;
    }
    default_device_params.device_reserve_memory_bytes =
        EnvUtil::getEnv("DEVICE_RESERVE_MEMORY_BYTES", default_device_params.device_reserve_memory_bytes);
    default_device_params.host_reserve_memory_bytes =
        EnvUtil::getEnv("HOST_RESERVE_MEMORY_BYTES", default_device_params.host_reserve_memory_bytes);
    ft::DeviceFactory::initDevices(global_params);
}

}  // namespace rtp_llm
