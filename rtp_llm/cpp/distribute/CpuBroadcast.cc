#include "rtp_llm/cpp/distribute/CpuBroadcast.h"

#include "rtp_llm/cpp/distribute/CpuBroadcaster.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <memory>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rtp_llm {
namespace {

std::unique_ptr<py::gil_scoped_release> releaseGilIfHeld() {
    if (!Py_IsInitialized() || !PyGILState_Check()) {
        return nullptr;
    }
    return std::make_unique<py::gil_scoped_release>();
}

}  // namespace

void execBroadcastCpu(const BroadcastParams& params, bool allow_fallback) {
    RTP_LLM_CHECK_WITH_INFO(
        params.root == 0, "execBroadcastCpu supports only root=0; got %ld", static_cast<long>(params.root));
    RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP,
                            "execBroadcastCpu supports only ParallelMode::TP; got %d",
                            static_cast<int>(params.mode));

    for (auto& t : params.buffers) {
        RTP_LLM_CHECK_WITH_INFO(
            t.is_cpu(), "execBroadcastCpu requires CPU tensors (got device=%s)", t.device().str().c_str());
    }

    auto& bcast = CpuBroadcaster::instance();
    if (!bcast.isInitialized()) {
        RTP_LLM_CHECK_WITH_INFO(allow_fallback, "execBroadcastCpu called before CpuBroadcaster is initialized");
        execBroadcast(params);
        execSyncCommunication(false);
        cudaSyncAndCheck();
        return;
    }
    for (auto& t : params.buffers) {
        auto contig = t.contiguous();
        {
            // Production engine workers do not hold the GIL, while direct
            // Python entry points may. Release it only when held so a stalled
            // peer cannot freeze unrelated Python threads.
            auto gil_release = releaseGilIfHeld();
            bcast.broadcast(contig.data_ptr(), contig.nbytes(), params.root);
        }
        if (!contig.is_same(t)) {
            t.copy_(contig);
        }
    }
}

}  // namespace rtp_llm
