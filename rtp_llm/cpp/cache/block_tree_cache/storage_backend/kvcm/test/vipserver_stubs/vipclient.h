#pragma once
#include <atomic>
#include "iphost.h"
#include "option.h"
namespace middleware::vipclient {
struct ApiState {
    std::atomic<int>  create{0}, init{0}, uninit{0}, destroy{0};
    std::atomic<bool> active{false};
};
inline ApiState& apiState() {
    // Keep observations alive while the production singleton is destroyed.
    static auto* state = new ApiState;
    return *state;
}
struct VipClientApi {
    static void CreateApi() {
        ++apiState().create;
    }
    static bool Init(const char*, const Option&) {
        if (++apiState().init == 1) {
            return false;
        }
        apiState().active = true;
        return true;
    }
    static void UnInit() {
        ++apiState().uninit;
        apiState().active = false;
    }
    static void DestoryApi() {
        ++apiState().destroy;
        apiState().active = false;
    }
    static bool QueryAllIp(const char* domain, IPHostArray* hosts, int) {
        if (!apiState().active) {
            return false;
        }
        hosts->hosts.push_back({domain});
        return true;
    }
};
}  // namespace middleware::vipclient
