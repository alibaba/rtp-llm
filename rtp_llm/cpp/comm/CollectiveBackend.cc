#include "rtp_llm/cpp/comm/CollectiveBackend.h"
#include "rtp_llm/cpp/comm/CollectiveBackendPybind.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <mutex>
#include <string>
#include <utility>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/python.h>

namespace rtp_llm {

namespace py = pybind11;

namespace {
std::mutex g_comm_mutex;

// Heap allocation avoids destroying Python objects from C++ static destructors
// after the interpreter has already been finalized.
py::function* g_broadcast_fn = nullptr;
py::function* g_allreduce_fn = nullptr;
py::function* g_allgather_fn = nullptr;

void clearCommOpsUnlocked() {
    py::function broadcast_fn;
    py::function allreduce_fn;
    py::function allgather_fn;
    if (g_broadcast_fn != nullptr) {
        broadcast_fn = std::move(*g_broadcast_fn);
        delete g_broadcast_fn;
        g_broadcast_fn = nullptr;
    }
    if (g_allreduce_fn != nullptr) {
        allreduce_fn = std::move(*g_allreduce_fn);
        delete g_allreduce_fn;
        g_allreduce_fn = nullptr;
    }
    if (g_allgather_fn != nullptr) {
        allgather_fn = std::move(*g_allgather_fn);
        delete g_allgather_fn;
        g_allgather_fn = nullptr;
    }
}
}  // anonymous namespace

void execBroadcast(const BroadcastParams& params) {
    RTP_LLM_CHECK_WITH_INFO(Py_IsInitialized(), "execBroadcast called before Python interpreter initialization");
    py::gil_scoped_acquire gil;
    py::function           fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_broadcast_fn != nullptr) {
            fn = *g_broadcast_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execBroadcast called but broadcast callback not registered via register_comm_ops");
    py::list tensors;
    for (auto& t : params.buffers)
        tensors.append(t);
    fn(tensors, params.root, static_cast<int>(params.mode));
}

void execBroadcastCpu(const BroadcastParams& params) {
    RTP_LLM_CHECK_WITH_INFO(
        params.root == 0, "execBroadcastCpu supports only root=0; got %ld", static_cast<long>(params.root));
    RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP,
                            "execBroadcastCpu supports only ParallelMode::TP; got %d",
                            static_cast<int>(params.mode));

    auto& broadcaster = CpuTpBroadcaster::instance();
    if (broadcaster.isInitialized()) {
        for (auto& tensor : params.buffers) {
            RTP_LLM_CHECK_WITH_INFO(
                tensor.is_cpu(), "execBroadcastCpu requires CPU tensors (got device=%s)", tensor.device().str().c_str());
            auto contiguous = tensor.contiguous();
            broadcaster.broadcast(contiguous.data_ptr(), contiguous.nbytes(), params.root);
            if (!contiguous.is_same(tensor)) {
                tensor.copy_(contiguous);
            }
        }
        return;
    }
    execBroadcast(params);
    execSyncCommunication(false);
    cudaSyncAndCheck();
}

bool isCpuTpBroadcasterInitialized() {
    return CpuTpBroadcaster::instance().isInitialized();
}

AllReduceOutput execAllReduce(const AllReduceParams& params) {
    RTP_LLM_CHECK_WITH_INFO(Py_IsInitialized(), "execAllReduce called before Python interpreter initialization");
    py::gil_scoped_acquire gil;
    py::function           fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_allreduce_fn != nullptr) {
            fn = *g_allreduce_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllReduce called but allreduce callback not registered via register_comm_ops");
    auto result = fn(params.buffer,
                     static_cast<int>(params.op),
                     static_cast<int>(params.mode),
                     params.dest.defined() ? py::cast(params.dest) : py::none());
    return AllReduceOutput{result.cast<torch::Tensor>()};
}

void execAllGather(const AllGatherParams& params) {
    RTP_LLM_CHECK_WITH_INFO(Py_IsInitialized(), "execAllGather called before Python interpreter initialization");
    py::gil_scoped_acquire gil;
    py::function           fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_allgather_fn != nullptr) {
            fn = *g_allgather_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllGather called but allgather callback not registered via register_comm_ops");
    py::list recv_list, send_list;
    for (auto& t : params.recv_buffers)
        recv_list.append(t);
    for (auto& t : params.send_buffers)
        send_list.append(t);
    fn(recv_list, static_cast<int>(params.mode), send_list, params.inplace);
}

void execSyncCommunication(bool timeout) {
    (void)timeout;  // Python ops are synchronous
}

void execSyncCommunication(ParallelMode mode, bool timeout) {
    (void)mode;
    (void)timeout;  // Python ops are synchronous
}

void registerCommPybindings(pybind11::module& m) {
    m.def(
        "register_comm_ops",
        [](py::function broadcast_fn, py::function allreduce_fn, py::function allgather_fn) {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            clearCommOpsUnlocked();
            g_broadcast_fn = new py::function(std::move(broadcast_fn));
            g_allreduce_fn = new py::function(std::move(allreduce_fn));
            g_allgather_fn = new py::function(std::move(allgather_fn));
        },
        py::arg("broadcast_fn"),
        py::arg("allreduce_fn"),
        py::arg("allgather_fn"),
        "Register Python callbacks for C++ communication ops.");

    m.def(
        "clear_comm_ops",
        []() {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            clearCommOpsUnlocked();
        },
        "Clear registered Python communication callbacks.");

    m.def(
        "init_cpu_tp_broadcaster",
        [](int tp_rank, int tp_size, const std::string& base_path) {
            py::gil_scoped_release release;
            CpuTpBroadcaster::instance().initialize(tp_rank, tp_size, base_path);
        },
        py::arg("tp_rank"),
        py::arg("tp_size"),
        py::arg("base_path"));

    m.def(
        "destroy_cpu_tp_broadcaster",
        []() {
            py::gil_scoped_release release;
            CpuTpBroadcaster::instance().reset();
        });
}

}  // namespace rtp_llm
