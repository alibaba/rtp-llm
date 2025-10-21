/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sstream>

#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm {

namespace {
int constexpr VOID_PTR_SZ = 2 + sizeof(void*) * 2;
}

RTPException::RTPException(char const* file, std::size_t line, const std::string& msg): std::runtime_error{""} {
    mNbFrames                 = backtrace(mCallstack.data(), MAX_FRAMES);
    auto const          trace = getTrace();
    std::runtime_error::operator=(
        std::runtime_error{rtp_llm::fmtstr("%s (%s:%zu)\n%s", msg.c_str(), file, line, trace.c_str())});
}

RTPException::~RTPException() noexcept = default;

std::string RTPException::getTrace() const {
    auto const         trace = backtrace_symbols(mCallstack.data(), mNbFrames);
    std::ostringstream buf;
    for (auto i = 1; i < mNbFrames; ++i) {
        Dl_info info;
        if (dladdr(mCallstack[i], &info) && info.dli_sname) {
            auto const clearName = demangle(info.dli_sname);
            buf << rtp_llm::fmtstr("%-3d %*p %s + %zd",
                                   i,
                                   VOID_PTR_SZ,
                                   mCallstack[i],
                                   clearName.c_str(),
                                   static_cast<char*>(mCallstack[i]) - static_cast<char*>(info.dli_saddr));
        } else {
            buf << rtp_llm::fmtstr("%-3d %*p %s", i, VOID_PTR_SZ, mCallstack[i], trace[i]);
        }
        if (i < mNbFrames - 1) {
            buf << std::endl;
        }
    }

    if (mNbFrames == MAX_FRAMES)
        buf << std::endl << "[truncated]";

    std::free(trace);
    return buf.str();
}

std::string RTPException::demangle(char const* name) {
    std::string clearName{name};
    auto        status    = -1;
    auto const  demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        clearName = demangled;
        std::free(demangled);
    }
    return clearName;
}

}  // namespace rtp_llm
