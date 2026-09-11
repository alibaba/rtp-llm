#include <gtest/gtest.h>
#include <limits>

#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

namespace rtp_llm {
namespace {
MMRdmaDescPB descriptor() {
    MMRdmaDescPB desc;
    desc.set_addr(4096);
    desc.set_handle("test-slot");
    desc.set_rdma_ip("127.0.0.1");
    desc.set_rdma_port(1234);
    desc.set_nbytes(32);
    desc.add_nic_rkeys()->set_rkey(1);
    auto* tensor = desc.add_tensors();
    tensor->set_role(MMRdmaTensorPB::EMBEDDING);
    tensor->set_data_type(TensorPB::FP32);
    tensor->add_shape(2);
    tensor->add_shape(4);
    tensor->set_nbytes(32);
    return desc;
}
}  // namespace

TEST(MMRdmaTransportTest, RejectsUnboundedOrInconsistentRemoteRanges) {
    auto desc = descriptor();
    EXPECT_TRUE(validateMMRdmaDescriptor(desc));
    desc.set_addr(std::numeric_limits<uint64_t>::max() - 16);
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
    desc = descriptor();
    desc.mutable_tensors(0)->set_offset(4);
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
    desc = descriptor();
    desc.mutable_tensors(0)->set_shape(0, std::numeric_limits<int64_t>::max());
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
    desc = descriptor();
    desc.mutable_tensors(0)->set_shape(0, -1);
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
    desc = descriptor();
    desc.add_tensors()->CopyFrom(desc.tensors(0));
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
    desc = descriptor();
    desc.clear_nic_rkeys();
    EXPECT_FALSE(validateMMRdmaDescriptor(desc));
}

TEST(MMRdmaTransportTest, GrpcAndMissingProviderModes) {
    // This target deliberately links only the interface, as an open-source build does.
    VitConfig config;
    EXPECT_EQ(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), nullptr);
    config.mm_transport_mode = "auto";
    EXPECT_EQ(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), nullptr);
    config.mm_transport_mode = "rdma";
    EXPECT_THROW(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), std::runtime_error);
    config.mm_transport_mode          = "auto";
    config.mm_rdma_max_inflight_bytes = 0;
    EXPECT_THROW(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), std::invalid_argument);
}
}  // namespace rtp_llm
