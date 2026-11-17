#pragma once

#include <cstdint>
#include <string>

#include <deep_jit/backend/ascend/driver.hpp>
#include <deep_jit/utils/exception.hpp>

namespace deep_jit::ascend {

struct DeviceProperties {
    std::string soc_name;
    int64_t num_aicpu_cores = 0;
    int64_t num_aicore_cores = 0;
    int64_t num_cube_cores = 0;
    int64_t num_vec_cores = 0;
    int64_t num_lanes_per_warp = 0;
    int64_t num_max_threads_per_vec_core = 0;
    int64_t num_ubuf_bytes_per_vec_core = 0;
    int64_t num_total_global_mem_bytes = 0;
    int64_t num_l2_cache_bytes = 0;
    int64_t npu_arch = 0;
};

class Device {
    DeviceProperties prop{};
    bool initialized = false;

public:
    const DeviceProperties& get_prop() {
        if (not initialized) {
            int device_index = 0;
            DJ_ACL_CHECK(driver::lazy_aclrtGetDevice(&device_index));
            const char* soc_name = driver::lazy_aclrtGetSocName();
            DJ_HOST_ASSERT(soc_name != nullptr, "aclrtGetSocName returned null");
            prop.soc_name = soc_name;

#define DJ_ASCEND_GET_PROP(name, attribute) \
            DJ_ACL_CHECK(driver::lazy_aclrtGetDeviceInfo(device_index, attribute, &prop.name))
            DJ_ASCEND_GET_PROP(num_aicpu_cores, ACL_DEV_ATTR_AICPU_CORE_NUM);
            DJ_ASCEND_GET_PROP(num_aicore_cores, ACL_DEV_ATTR_AICORE_CORE_NUM);
            DJ_ASCEND_GET_PROP(num_cube_cores, ACL_DEV_ATTR_CUBE_CORE_NUM);
            DJ_ASCEND_GET_PROP(num_vec_cores, ACL_DEV_ATTR_VECTOR_CORE_NUM);
            DJ_ASCEND_GET_PROP(num_lanes_per_warp, ACL_DEV_ATTR_WARP_SIZE);
            DJ_ASCEND_GET_PROP(num_max_threads_per_vec_core, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE);
            DJ_ASCEND_GET_PROP(num_ubuf_bytes_per_vec_core, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE);
            DJ_ASCEND_GET_PROP(num_total_global_mem_bytes, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE);
            DJ_ASCEND_GET_PROP(num_l2_cache_bytes, ACL_DEV_ATTR_L2_CACHE_SIZE);
            DJ_ASCEND_GET_PROP(npu_arch, ACL_DEV_ATTR_NPU_ARCH);
#undef DJ_ASCEND_GET_PROP
            initialized = true;
        }
        return prop;
    }

    int get_npu_arch() { return static_cast<int>(get_prop().npu_arch); }

    int get_num_aicore_cores() { return static_cast<int>(get_prop().num_aicore_cores); }

    int get_num_sms() { return get_num_aicore_cores(); }

    int get_num_vec_cores() { return static_cast<int>(get_prop().num_vec_cores); }

    int get_num_cube_cores() { return static_cast<int>(get_prop().num_cube_cores); }

    int get_num_vec_cores_per_ai_core() {
        const auto num_ai_cores = get_num_aicore_cores();
        const auto num_vector_cores = get_num_vec_cores();
        DJ_HOST_ASSERT(num_ai_cores > 0 and num_vector_cores % num_ai_cores == 0,
                       "Ascend vector core count must be divisible by AI core count");
        return num_vector_cores / num_ai_cores;
    }

    int64_t get_num_l2_cache_bytes() { return get_prop().num_l2_cache_bytes; }

    int64_t get_num_ubuf_bytes_per_vec_core() { return get_prop().num_ubuf_bytes_per_vec_core; }
};

}  // namespace deep_jit::ascend
