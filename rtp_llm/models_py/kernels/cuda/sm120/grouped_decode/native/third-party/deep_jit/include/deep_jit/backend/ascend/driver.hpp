#pragma once

#include <acl/acl.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_jit::ascend::driver {

DJ_DECL_LAZY_DL_HANDLE(get_acl_handle, "libascendcl.so");

DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclGetRecentErrMsg);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtBinaryLoadFromFile);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtBinaryUnLoad);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtBinaryGetFunction);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtLaunchKernelWithHostArgs);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtGetDevice);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtGetSocName);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtGetDeviceInfo);
DJ_DECL_LAZY_DL_FUNCTION(get_acl_handle, aclrtSynchronizeDevice);

inline void check_acl(const aclError error, const char* expression) {
    if (error == ACL_SUCCESS)
        return;

    const char* message = lazy_aclGetRecentErrMsg();
    DJ_PANIC("{} failed with ACL error {}: {}",
             expression,
             static_cast<int>(error),
             message == nullptr ? "unknown" : message);
}

#ifndef DJ_ACL_CHECK
#define DJ_ACL_CHECK(expr) ::deep_jit::ascend::driver::check_acl((expr), #expr)
#endif

}  // namespace deep_jit::ascend::driver
