#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include <string>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(PdKvWritebackManagerTest, DisabledConfigSkipsLaunch) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = false;
    PdKvWritebackManager manager(pd_config, nullptr);

    PdKvWritebackLaunchRequest request;
    request.manifest.reusable_block_count = 1;

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Skipped);
    EXPECT_EQ(result.reason, "disabled");
}

TEST(PdKvWritebackManagerTest, RejectsIncompatibleSeqSize) {
    PdKvWritebackCompatibility source;
    source.seq_size_per_block = 16;
    PdKvWritebackCompatibility destination;
    destination.seq_size_per_block = 32;

    auto status = validatePdKvWritebackCompatibility(source, destination);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(std::string(status.message()).find("seq_size_per_block"), std::string::npos);
}

TEST(PdKvWritebackManagerTest, EnabledCompatibleLaunchStarts) {
    PDSepConfig pd_config;
    pd_config.enable_pd_kv_cache_writeback = true;
    PdKvWritebackManager manager(pd_config, nullptr);

    PdKvWritebackLaunchRequest request;
    request.manifest.reusable_block_count = 2;
    request.source.seq_size_per_block     = 16;
    request.source.layer_count            = 4;
    request.source.group_count            = 1;
    request.source.partition_count        = 1;
    request.source.layer_to_group_id      = {0, 0, 0, 0};
    request.source.group_types            = {0};
    request.destination                   = request.source;
    request.source_prefill_grpc_addrs     = {"127.0.0.1:9000"};

    auto result = manager.launchFromDecode(request);

    EXPECT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
}

}  // namespace
}  // namespace rtp_llm
