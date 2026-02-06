#include "absl/debugging/symbolize.h"

#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <execinfo.h>
#include <unistd.h>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <dlfcn.h>
#include <cxxabi.h>
#include <memory>
#include <cstdio>
#include <array>

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace rtp_llm {

static constexpr int kMaxStackDepth = 64;

// Python stack trace related constants
static constexpr const char* kSitePackages       = "site-packages";
static constexpr const char* kRtpLlmPath         = "RTP-LLM";
static constexpr size_t      kMaxTraceLineLength = 50;
static constexpr size_t      kMaxTraceLines      = 5;
static constexpr size_t      kHeadTraceLines     = 2;
static constexpr size_t      kTailTraceLines     = 2;
// Helper function to resolve symlink to actual file path

static std::string resolvePath(const char* path) {
    if (!path || strlen(path) == 0) {
        return "";
    }
    char    resolved[1024];
    ssize_t len = readlink(path, resolved, sizeof(resolved) - 1);
    if (len != -1) {
        resolved[len] = '\0';
        return std::string(resolved);
    }
    return std::string(path);
}

// Helper function to get source file and line number using addr2line
static std::string getSourceLocation(const void* addr, const char* binary_path, const void* base_addr) {
    if (!binary_path || strlen(binary_path) == 0) {
        return "";
    }

    // Resolve symlink to actual path (important for Bazel runfiles)
    std::string actual_path = resolvePath(binary_path);
    if (actual_path.empty()) {
        actual_path = binary_path;
    }

    // Calculate offset from base address (addr2line needs offset, not absolute address)
    uintptr_t offset = 0;
    if (base_addr) {
        offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(base_addr);
    } else {
        // If no base address, use absolute address (may work for main executable)
        offset = reinterpret_cast<uintptr_t>(addr);
    }

    char offset_str[32];
    snprintf(offset_str, sizeof(offset_str), "0x%lx", offset);

    std::array<char, 1024> cmd;
    snprintf(cmd.data(), cmd.size(), "addr2line -e %s -f -C -i %s 2>/dev/null", actual_path.c_str(), offset_str);

    FILE* pipe = popen(cmd.data(), "r");
    if (!pipe) {
        return "";
    }

    std::string result;
    char        buffer[512];
    int         line_count = 0;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        // Remove trailing newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        line_count++;
        if (line_count == 1) {
            // First line is function name, skip it (we already have it from dladdr)
            continue;
        } else if (line_count == 2) {
            // Second line is file:line
            if (strlen(buffer) > 0 && strcmp(buffer, "??:0") != 0 && strcmp(buffer, "??") != 0) {
                result = buffer;
            }
            break;
        }
    }
    int status = pclose(pipe);
    // If addr2line failed (non-zero exit), return empty string
    if (status != 0) {
        return "";
    }
    return result;
}

std::string getStackTrace() {
    std::ostringstream stack_ss;
    void*              addrs[kMaxStackDepth];

    int stack_depth = backtrace(addrs, kMaxStackDepth);

    // Use detailed formatting with dladdr and demangling for better symbol resolution
    char**                                  symlist = backtrace_symbols(addrs, stack_depth);
    std::unique_ptr<char*, decltype(&free)> symlist_guard(symlist, free);

    for (int i = 2; i < stack_depth; ++i) {
        void*   addr = addrs[i];
        Dl_info info;
        bool    dladdr_success = (dladdr(addr, &info) != 0);

        stack_ss << "  [" << std::setw(2) << std::setfill('0') << i << "] ";

        if (dladdr_success && info.dli_sname) {
            int   status    = 0;
            char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
            std::unique_ptr<char, decltype(&free)> demangled_guard(demangled, free);

            const char*     func_name = (status == 0 && demangled) ? demangled : info.dli_sname;
            const ptrdiff_t offset    = static_cast<char*>(addr) - static_cast<char*>(info.dli_saddr);

            // Try absl::Symbolize for better symbol resolution
            char absl_symbol[1024];
            if (absl::Symbolize(addr, absl_symbol, sizeof(absl_symbol))) {
                stack_ss << absl_symbol << " in " << (info.dli_fname ? info.dli_fname : "???");
            } else {
                stack_ss << func_name << "+0x" << std::hex << offset << std::dec << " in "
                         << (info.dli_fname ? info.dli_fname : "???");
            }
        } else if (symlist && symlist[i]) {
            stack_ss << symlist[i];
        } else {
            stack_ss << "??? (addr: " << addr << ")";
        }

        // Always try to get source file and line number if we have binary path
        // This ensures we get line numbers even when symbol resolution fails
        if (dladdr_success && info.dli_fname) {
            std::string source_loc = getSourceLocation(addr, info.dli_fname, info.dli_fbase);
            if (!source_loc.empty()) {
                stack_ss << " at " << source_loc;
            }
        } else if (symlist && symlist[i]) {
            // Try to extract binary path from backtrace_symbols output and get line number
            // Format: /path/to/binary(function+offset) [address]
            std::string sym_str   = symlist[i];
            size_t      paren_pos = sym_str.find('(');
            if (paren_pos != std::string::npos) {
                std::string binary_path = sym_str.substr(0, paren_pos);
                // Try dladdr again to get base address
                if (dladdr(addr, &info) && info.dli_fname) {
                    std::string source_loc = getSourceLocation(addr, info.dli_fname, info.dli_fbase);
                    if (!source_loc.empty()) {
                        stack_ss << " at " << source_loc;
                    }
                } else {
                    // Try with the extracted path (may be a symlink)
                    std::string source_loc = getSourceLocation(addr, binary_path.c_str(), nullptr);
                    if (!source_loc.empty()) {
                        stack_ss << " at " << source_loc;
                    }
                }
            }
        }

        stack_ss << "\n";
    }

    return stack_ss.str();
}

void printStackTrace() {
    RTP_LLM_STACKTRACE_LOG_INFO("%s", getStackTrace().c_str());
    fflush(stdout);
    fflush(stderr);
}

// Extract and format original Python stack frames with code lines
static std::vector<std::string> getOriginPythonStack() {
    std::vector<std::string> frames;

    if (!Py_IsInitialized()) {
        return frames;
    }

    py::gil_scoped_acquire gil;

    PyThreadState* tstate = PyThreadState_GET();
    if (!tstate) {
        return frames;
    }

    PyFrameObject* frame = PyThreadState_GetFrame(tstate);
    if (!frame) {
        return frames;
    }

    py::module_ traceback  = py::module_::import("traceback");
    py::object  py_frame   = py::reinterpret_borrow<py::object>((PyObject*)frame);
    py::list    stack_list = traceback.attr("extract_stack")(py_frame);
    Py_DECREF(frame);

    for (auto item : stack_list) {
        py::object filename = item.attr("filename");
        py::object lineno   = item.attr("lineno");
        py::object name     = item.attr("name");
        py::object line     = item.attr("line");

        std::string full_path = py::str(filename).cast<std::string>();

        // Simplify file path: start from the level after site-packages or RTP-LLM
        std::string short_path        = full_path;
        size_t      site_packages_pos = full_path.find(kSitePackages);
        size_t      rtp_llm_pos       = full_path.find(kRtpLlmPath);

        if (site_packages_pos != std::string::npos) {
            size_t next_slash = full_path.find('/', site_packages_pos + strlen(kSitePackages));
            if (next_slash != std::string::npos) {
                short_path = full_path.substr(next_slash + 1);
            }
        } else if (rtp_llm_pos != std::string::npos) {
            size_t next_slash = full_path.find('/', rtp_llm_pos + strlen(kRtpLlmPath));
            if (next_slash != std::string::npos) {
                short_path = full_path.substr(next_slash + 1);
            }
        }

        std::stringstream frame_ss;
        frame_ss << "    " << short_path << ":" << py::str(lineno).cast<std::string>() << " in "
                 << py::str(name).cast<std::string>() << "\n";

        if (!line.is_none()) {
            std::string code_line = py::str(line).cast<std::string>();
            size_t      start     = code_line.find_first_not_of(" \t");
            size_t      end       = code_line.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                code_line = code_line.substr(start, end - start + 1);
            }
            if (code_line.length() > kMaxTraceLineLength) {
                code_line = code_line.substr(0, kMaxTraceLineLength - 3) + "...";
            }
            frame_ss << "    └─ " << code_line << "\n";
        }
        frames.push_back(frame_ss.str());
    }

    return frames;
}

// Filter and reformat Python stack frames with box drawing characters
static std::string reformatPythonStack(const std::vector<std::string>& frames) {
    if (frames.empty()) {
        return "No relevant Python stack trace available";
    }

    // Filter stack frames to keep relevant ones
    std::vector<std::string> filtered_frames;
    int                      model_desc_index = -1;

    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].find("model_desc") != std::string::npos) {
            model_desc_index = i;
            break;
        }
    }

    size_t start_index = 0;
    if (model_desc_index >= 0) {
        start_index = model_desc_index;
    } else if (frames.size() > kMaxTraceLines) {
        start_index = frames.size() - kMaxTraceLines;
    }

    for (size_t i = start_index; i < frames.size(); ++i) {
        const std::string& frame = frames[i];
        if (frame.find("torch/nn/modules/module.py") != std::string::npos) {
            continue;
        }
        filtered_frames.push_back(frame);
    }

    // If still too many frames, keep only head and tail parts
    std::vector<std::string> final_frames;
    if (filtered_frames.size() > kMaxTraceLines) {
        for (size_t i = 0; i < kHeadTraceLines; ++i) {
            final_frames.push_back(filtered_frames[i]);
        }
        final_frames.push_back("    ...\n");
        for (size_t i = filtered_frames.size() - kTailTraceLines; i < filtered_frames.size(); ++i) {
            final_frames.push_back(filtered_frames[i]);
        }
    } else {
        final_frames = filtered_frames;
    }

    if (final_frames.empty()) {
        return "No relevant Python stack trace available";
    }

    // Format output with box drawing characters
    std::stringstream stack_ss;
    stack_ss << "\n  ┌─ Python Stack Trace ─────────────────────────────────────────\n";
    for (const auto& frame : final_frames) {
        std::istringstream iss(frame);
        std::string        line;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                stack_ss << "  │" << line << "\n";
            }
        }
    }
    stack_ss << "  └──────────────────────────────────────────────────────────────\n";

    return stack_ss.str();
}

std::string getPythonStackTrace() {
    if (!Py_IsInitialized()) {
        return "Python interpreter not initialized";
    }

    try {
        std::vector<std::string> frames = getOriginPythonStack();
        return reformatPythonStack(frames);
    } catch (py::error_already_set& e) {
        std::stringstream err_ss;
        err_ss << "Error while extracting Python stack trace: " << e.what();
        e.restore();
        PyErr_Clear();
        return err_ss.str();
    } catch (const std::exception& e) {
        std::stringstream err_ss;
        err_ss << "Error while extracting Python stack trace: " << e.what();
        return err_ss.str();
    } catch (...) {
        return "Unknown error while extracting Python stack trace";
    }
}

}  // namespace rtp_llm
