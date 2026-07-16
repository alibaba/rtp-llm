#include "rtp_llm/cpp/comm/CollectiveBackend.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <mutex>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace rtp_llm {

namespace py = pybind11;

namespace {
std::mutex   g_comm_mutex;
py::function g_broadcast_fn;  // (tensors: list[Tensor], root: int, mode: int) -> None
py::function g_allreduce_fn;  // (tensor: Tensor, op: int, mode: int, dest: Optional[Tensor]) -> Tensor
py::function
    g_allgather_fn;  // (recv_buffers: list[Tensor], mode: int, send_buffers: list[Tensor], inplace: bool) -> None
}  // anonymous namespace

void execBroadcast(const BroadcastParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_broadcast_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execBroadcast called but broadcast callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    py::list               tensors;
    for (auto& t : params.buffers)
        tensors.append(t);
    fn(tensors, params.root, static_cast<int>(params.mode));
}

AllReduceOutput execAllReduce(const AllReduceParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_allreduce_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllReduce called but allreduce callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    auto                   result = fn(params.buffer,
                     static_cast<int>(params.op),
                     static_cast<int>(params.mode),
                     params.dest.defined() ? py::cast(params.dest) : py::none());
    return AllReduceOutput{result.cast<torch::Tensor>()};
}

void execAllGather(const AllGatherParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_allgather_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllGather called but allgather callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    py::list               recv_list, send_list;
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
            g_broadcast_fn = std::move(broadcast_fn);
            g_allreduce_fn = std::move(allreduce_fn);
            g_allgather_fn = std::move(allgather_fn);
        },
        py::arg("broadcast_fn"),
        py::arg("allreduce_fn"),
        py::arg("allgather_fn"),
        "Register Python callbacks for C++ communication ops.");

    m.def(
        "clear_comm_ops",
        []() {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            g_broadcast_fn = py::function();
            g_allreduce_fn = py::function();
            g_allgather_fn = py::function();
        },
        "Clear registered Python communication callbacks.");
}

}  // namespace rtp_llm
