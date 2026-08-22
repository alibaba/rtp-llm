#include <gtest/gtest.h>
#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/config/VitConfigExtract.h"

// Exercise the real Python-to-C++ mapping with a live interpreter.
namespace py = pybind11;

namespace rtp_llm {

namespace {

py::object pythonVitConfigWithDistinctValues() {
    py::object config = py::module_::import("rtp_llm.config.py_config_modules").attr("VitConfig")();
    config.attr("vit_separation") = 2;  // VIT_SEPARATION_REMOTE
    py::object transport = config.attr("output_transport");
    py::object control = transport.attr("control");
    py::object rdma = transport.attr("rdma");
    transport.attr("mode") = "grpc";
    control.attr("release_timeout_ms") = 456;
    rdma.attr("bind_ip") = "10.1.2.3";
    rdma.attr("port") = 12345;
    rdma.attr("connect_timeout_ms") = 234;
    rdma.attr("read_timeout_ms") = 3456;
    rdma.attr("slot_gc_timeout_ms") = 5678;
    rdma.attr("max_inflight_bytes") = 6789;
    rdma.attr("max_slot_bytes") = 7890;
    return config;
}

}  // namespace

TEST(VitConfigExtractTest, CopiesEveryFieldIntoItsOwnMember) {
    const VitConfig cfg = extractVitConfig(pythonVitConfigWithDistinctValues());
    const auto&     transport = cfg.output_transport;
    const auto&     control   = transport.control;
    const auto&     rdma      = transport.rdma;

    EXPECT_EQ(cfg.vit_separation, VitSeparation::VIT_SEPARATION_REMOTE);
    EXPECT_EQ(transport.mode, "grpc");
    EXPECT_EQ(rdma.bind_ip, "10.1.2.3");
    EXPECT_EQ(rdma.port, 12345);
    EXPECT_EQ(rdma.connect_timeout_ms, 234);
    EXPECT_EQ(rdma.read_timeout_ms, 3456);
    EXPECT_EQ(control.release_timeout_ms, 456);
    EXPECT_EQ(rdma.slot_gc_timeout_ms, 5678);
    EXPECT_EQ(rdma.max_inflight_bytes, 6789);
    EXPECT_EQ(rdma.max_slot_bytes, 7890);
}

TEST(VitConfigExtractTest, NoneYieldsTheCppDefaults) {
    const VitConfig defaults;
    const VitConfig cfg = extractVitConfig(py::none());

    EXPECT_EQ(cfg.vit_separation, defaults.vit_separation);
    EXPECT_EQ(cfg.output_transport.mode, defaults.output_transport.mode);
    EXPECT_EQ(cfg.output_transport.control.release_timeout_ms,
              defaults.output_transport.control.release_timeout_ms);
    EXPECT_EQ(cfg.output_transport.rdma.read_timeout_ms, defaults.output_transport.rdma.read_timeout_ms);
    EXPECT_EQ(cfg.output_transport.rdma.max_slot_bytes, defaults.output_transport.rdma.max_slot_bytes);
}

TEST(VitConfigExtractTest, MissingFieldIsAConfigurationError) {
    py::object incomplete = py::module_::import("types").attr("SimpleNamespace")();
    incomplete.attr("vit_separation") = 0;

    EXPECT_THROW(extractVitConfig(incomplete), py::error_already_set);
}

}  // namespace rtp_llm
