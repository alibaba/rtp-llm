#include "rtp_llm/cpp/comm/CollectiveBackend.h"
#include "rtp_llm/cpp/comm/CollectiveBackendPybind.h"

#include <gtest/gtest.h>
#include <functional>
#include <pybind11/embed.h>
#include <string>
#include <vector>

namespace rtp_llm {
namespace {

namespace py = pybind11;

void expectError(const std::function<void()>& invoke, const std::string& operation, const std::string& detail) {
    try {
        invoke();
        FAIL() << operation << " unexpectedly succeeded";
    } catch (const std::exception& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find(operation), std::string::npos);
        EXPECT_NE(message.find(detail), std::string::npos);
    }
}

TEST(CollectiveBackendTest, EnforcesCallbackLifecycle) {
    std::vector<torch::Tensor> buffers;
    BroadcastParams            broadcast{buffers, 0};
    AllReduceParams            allreduce{torch::Tensor(), ReduceOp::Sum};
    AllGatherParams            allgather{buffers};

    expectError([&]() { execBroadcast(broadcast); }, "execBroadcast", "before Python interpreter initialization");
    expectError([&]() { (void)execAllReduce(allreduce); }, "execAllReduce", "before Python interpreter initialization");
    expectError([&]() { execAllGather(allgather); }, "execAllGather", "before Python interpreter initialization");

    py::scoped_interpreter interpreter;
    py::module_            module = py::module_::import("__main__");
    registerCommPybindings(module);

    bool broadcast_called = false;
    auto broadcast_fn     = py::cpp_function([&](py::list, int64_t, int) { broadcast_called = true; });
    auto unused_fn        = py::cpp_function([](py::args) { return py::none(); });
    module.attr("register_comm_ops")(broadcast_fn, unused_fn, unused_fn);

    execBroadcast(broadcast);
    EXPECT_TRUE(broadcast_called);

    module.attr("clear_comm_ops")();
    expectError([&]() { execBroadcast(broadcast); }, "execBroadcast", "callback not registered");
}

}  // namespace
}  // namespace rtp_llm
