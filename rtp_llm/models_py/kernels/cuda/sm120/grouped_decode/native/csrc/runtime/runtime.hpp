#pragma once
#include <deep_jit/utils/lazy.hpp>
#include "jit.hpp"
namespace deep_gemm {
class Runtime {
public:
    int tc_util = 0;

    int get_num_sms() {
        return jit->device.get_num_sms();
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

};

inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });

}  // namespace deep_gemm
