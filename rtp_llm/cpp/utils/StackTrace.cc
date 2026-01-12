#include "absl/debugging/symbolize.h"

#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <execinfo.h>
#include <unistd.h>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>

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

std::string getStackTrace() {
    std::stringstream stack_ss;
    void*             addrs[kMaxStackDepth];

    int stack_depth = backtrace(addrs, kMaxStackDepth);
    for (int i = 2; i < stack_depth; ++i) {
        char line[2048];
        char buf[1024];
        if (absl::Symbolize(addrs[i], buf, sizeof(buf))) {
            snprintf(line, 2048, "@  %16p  %s\n", addrs[i], buf);
        } else {
            snprintf(line, 2048, "@  %16p  (unknown)\n", addrs[i]);
        }
        stack_ss << std::string(line);
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
