#pragma once

#if RTP_DG120_ENABLE_LIBDW
#include <elfutils/libdwfl.h>
#endif

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <exception>
#include <execinfo.h>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <unistd.h>

namespace deep_jit::exception {

namespace detail {

struct DwflApi {
    struct FrameInfo {
        const char* function = nullptr;
        std::string file_line = "??:0";
    };

#if RTP_DG120_ENABLE_LIBDW
    void* handle = nullptr;
    decltype(&::dwfl_begin) dwfl_begin = nullptr;
    decltype(&::dwfl_linux_proc_report) dwfl_linux_proc_report = nullptr;
    decltype(&::dwfl_report_end) dwfl_report_end = nullptr;
    decltype(&::dwfl_addrmodule) dwfl_addrmodule = nullptr;
    decltype(&::dwfl_module_addrname) dwfl_module_addrname = nullptr;
    decltype(&::dwfl_module_getsrc) dwfl_module_getsrc = nullptr;
    decltype(&::dwfl_lineinfo) dwfl_lineinfo = nullptr;
    decltype(&::dwfl_linux_proc_find_elf) dwfl_linux_proc_find_elf = nullptr;

    Dwfl_Callbacks callbacks = {};
    Dwfl* dwfl = nullptr;

    DwflApi() {
        handle = ::dlopen("libdw.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) return;

#define DJ_LOAD_DWFL_FUNCTION(name) \
        name = reinterpret_cast<decltype(name)>(::dlsym(handle, #name))

        DJ_LOAD_DWFL_FUNCTION(dwfl_begin);
        DJ_LOAD_DWFL_FUNCTION(dwfl_linux_proc_report);
        DJ_LOAD_DWFL_FUNCTION(dwfl_report_end);
        DJ_LOAD_DWFL_FUNCTION(dwfl_addrmodule);
        DJ_LOAD_DWFL_FUNCTION(dwfl_module_addrname);
        DJ_LOAD_DWFL_FUNCTION(dwfl_module_getsrc);
        DJ_LOAD_DWFL_FUNCTION(dwfl_lineinfo);
        DJ_LOAD_DWFL_FUNCTION(dwfl_linux_proc_find_elf);

#undef DJ_LOAD_DWFL_FUNCTION

        callbacks.find_elf = dwfl_linux_proc_find_elf;
        callbacks.find_debuginfo = +[](
            Dwfl_Module*, void**, const char*,
            Dwarf_Addr, const char*, const char*,
            GElf_Word, char**
        ) {
            return -1;
        };
        dwfl = dwfl_begin(&callbacks);
    }

    bool valid() const {
        return handle != nullptr and dwfl != nullptr;
    }

    void refresh() const {
        if (not valid())
            return;
        dwfl_linux_proc_report(dwfl, ::getpid());
        dwfl_report_end(dwfl, nullptr, nullptr);
    }

    FrameInfo find(const Dwarf_Addr address) const {
        FrameInfo result;
        if (not valid())
            return result;
        Dwfl_Module* module = dwfl_addrmodule(dwfl, address);
        if (module == nullptr)
            return result;

        result.function = dwfl_module_addrname(module, address);
        Dwfl_Line* line = dwfl_module_getsrc(module, address);
        if (line != nullptr) {
            int line_number = 0;
            const char* file = dwfl_lineinfo(line, nullptr, &line_number, nullptr, nullptr, nullptr);
            result.file_line = fmt::format("{}:{}", file == nullptr ? "??" : file, line_number);
        }
        return result;
    }
#else
    void refresh() const {}
    FrameInfo find(std::uintptr_t) const { return {}; }
#endif
};

inline bool is_python_frame(const Dl_info& info) {
    if (info.dli_sname != nullptr) {
        const std::string_view symbol(info.dli_sname);
        if (symbol.starts_with("Py") or symbol.starts_with("_Py"))
            return true;
    }
    if (info.dli_fname == nullptr)
        return false;

    std::string_view filename(info.dli_fname);
    if (const auto slash = filename.find_last_of('/'); slash != std::string_view::npos)
        filename.remove_prefix(slash + 1);
    return filename == "python" or filename.starts_with("python3") or filename.starts_with("libpython");
}

inline std::string demangle(const char* name) {
    if (name == nullptr)
        return "<unknown function>";
    int status = 0;
    char* value = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string result = status == 0 and value != nullptr ? value : name;
    std::free(value);
    return result;
}

template <std::size_t num_frames_to_skip = 0>
inline __attribute__((noinline)) std::string get_backtrace() {
    static DwflApi api;
    api.refresh();

    constexpr int max_num_frames = 16;
    std::array<void*, max_num_frames + num_frames_to_skip> frames = {};
    const int num_frames = ::backtrace(frames.data(), static_cast<int>(frames.size()));
    const int first_frame = static_cast<int>(num_frames_to_skip);
    std::string result;
    for (int i = 0; i < max_num_frames; ++i) {
        if (first_frame + i >= num_frames)
            break;
        const auto address = reinterpret_cast<std::uintptr_t>(frames[first_frame + i]);
        const auto lookup_address = address == 0 ? 0 : address - 1;
        Dl_info info = {};
        ::dladdr(reinterpret_cast<void*>(lookup_address), &info);
        if (is_python_frame(info))
            break;

        const auto dwfl_info = api.find(lookup_address);
        const char* function = dwfl_info.function == nullptr ? info.dli_sname : dwfl_info.function;
        const auto library_base = reinterpret_cast<std::uintptr_t>(info.dli_fbase);
        result += fmt::format("  #{} {}+{:#x} {} at {}\n",
                              i,
                              info.dli_fname == nullptr ? "??" : info.dli_fname,
                              lookup_address - library_base,
                              demangle(function),
                              dwfl_info.file_line);
    }
    return result;
}

} // namespace detail

class Exception final : public std::exception {
public:
    std::string message = {};

    explicit __attribute__((noinline)) Exception(
        const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
        // use `-g1` compile option to enable line number information in backtrace
        const auto trace = detail::get_backtrace<2>();
        if (not trace.empty())
            message += "\nC++ trace (most recent call first):\n" + trace;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

} // namespace deep_jit::exception

#ifndef DJ_STATIC_ASSERT
#define DJ_STATIC_ASSERT(cond, ...) static_assert(cond, "" __VA_ARGS__)
#endif

#ifndef DJ_HOST_ASSERT
#define DJ_HOST_ASSERT(cond, ...) \
do { \
    if (not (cond)) { \
        throw deep_jit::exception::Exception("Assertion", __FILE__, __LINE__, std::string(#cond) __VA_OPT__(+ ": " + fmt::format(__VA_ARGS__))); \
    } \
} while (0)
#endif

#ifndef DJ_PANIC
#define DJ_PANIC(...) \
do { \
    throw deep_jit::exception::Exception("Panic", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
} while (0)
#endif
