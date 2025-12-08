#include <gtest/gtest.h>

#include "rtp_llm/cpp/disaggregate/p2p_connector/AsymmetricTpUtil.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

namespace rtp_llm {

class AsymmetricTpUtilTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 默认的 GptInitParameter 设置
        gpt_init_parameter_ = GptInitParameter();
    }

    void TearDown() override {}

protected:
    GptInitParameter gpt_init_parameter_;
};

// 场景：tp_size=4, decode_transfer_servers=2, 每个 decode 接收 2 个 prefill 的数据
TEST_F(AsymmetricTpUtilTest, HandleNP1D_2_4) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {
        {"192.168.1.10", 8080},
        {"192.168.1.11", 8080},
    };

    std::vector<std::vector<AsymmetricTPContext>> expected_contexts = {
        {
            {"192.168.1.10", 8080, 1, 0, 2, 0},
        },
        {
            {"192.168.1.10", 8080, 1, 0, 2, 1},
        },
        {
            {"192.168.1.11", 8080, 1, 0, 2, 0},
        },
        {
            {"192.168.1.11", 8080, 1, 0, 2, 1},
        },
    };

    for (int tp_rank = 0; tp_rank < 4; ++tp_rank) {
        GptInitParameter gpt_init_parameter = gpt_init_parameter_;
        gpt_init_parameter.tp_rank_         = tp_rank;
        gpt_init_parameter.tp_size_         = 4;

        AsymmetricTpUtil util(gpt_init_parameter);

        auto contexts = util.handleAsymmetricTP(decode_transfer_servers);
        ASSERT_EQ(contexts.size(), expected_contexts[tp_rank].size());
        for (size_t i = 0; i < contexts.size(); ++i) {
            EXPECT_EQ(contexts[i].decode_ip, expected_contexts[tp_rank][i].decode_ip);
            EXPECT_EQ(contexts[i].decode_port, expected_contexts[tp_rank][i].decode_port);
            EXPECT_EQ(contexts[i].local_partition_count, expected_contexts[tp_rank][i].local_partition_count);
            EXPECT_EQ(contexts[i].local_partition_id, expected_contexts[tp_rank][i].local_partition_id);
            EXPECT_EQ(contexts[i].remote_partition_count, expected_contexts[tp_rank][i].remote_partition_count);
            EXPECT_EQ(contexts[i].remote_partition_id, expected_contexts[tp_rank][i].remote_partition_id);
        }
    }
}

// 测试 4 decode -> 2 prefill
TEST_F(AsymmetricTpUtilTest, HandleND1P_4_2) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {
        {"192.168.1.10", 8080},
        {"192.168.1.11", 8080},
        {"192.168.1.12", 8080},
        {"192.168.1.13", 8080},
    };

    std::vector<std::vector<AsymmetricTPContext>> expected_contexts = {
        {
            {"192.168.1.10", 8080, 2, 0, 1, 0},
            {"192.168.1.11", 8080, 2, 1, 1, 0},
        },
        {
            {"192.168.1.12", 8080, 2, 0, 1, 0},
            {"192.168.1.13", 8080, 2, 1, 1, 0},
        },
    };

    for (int tp_rank = 0; tp_rank < 2; ++tp_rank) {
        GptInitParameter gpt_init_parameter = gpt_init_parameter_;
        gpt_init_parameter.tp_rank_         = tp_rank;
        gpt_init_parameter.tp_size_         = 2;
        AsymmetricTpUtil util(gpt_init_parameter);
        auto             contexts = util.handleAsymmetricTP(decode_transfer_servers);
        ASSERT_EQ(contexts.size(), expected_contexts[tp_rank].size());

        for (size_t i = 0; i < contexts.size(); ++i) {
            EXPECT_EQ(contexts[i].decode_ip, expected_contexts[tp_rank][i].decode_ip);
            EXPECT_EQ(contexts[i].decode_port, expected_contexts[tp_rank][i].decode_port);
            EXPECT_EQ(contexts[i].local_partition_count, expected_contexts[tp_rank][i].local_partition_count);
            EXPECT_EQ(contexts[i].local_partition_id, expected_contexts[tp_rank][i].local_partition_id);
            EXPECT_EQ(contexts[i].remote_partition_count, expected_contexts[tp_rank][i].remote_partition_count);
            EXPECT_EQ(contexts[i].remote_partition_id, expected_contexts[tp_rank][i].remote_partition_id);
        }
    }
}

TEST_F(AsymmetricTpUtilTest, HandleNPND_2_2) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {
        {"192.168.1.10", 8080},
        {"192.168.1.11", 8080},
    };

    std::vector<std::vector<AsymmetricTPContext>> expected_contexts = {
        {
            {"192.168.1.10", 8080, 1, 0, 1, 0},
        },
        {
            {"192.168.1.11", 8080, 1, 0, 1, 0},
        },
    };

    for (int tp_rank = 0; tp_rank < 2; ++tp_rank) {
        GptInitParameter gpt_init_parameter = gpt_init_parameter_;
        gpt_init_parameter.tp_rank_         = tp_rank;
        gpt_init_parameter.tp_size_         = 2;
        AsymmetricTpUtil util(gpt_init_parameter);
        auto             contexts = util.handleAsymmetricTP(decode_transfer_servers);
        ASSERT_EQ(contexts.size(), expected_contexts[tp_rank].size());
        for (size_t i = 0; i < contexts.size(); ++i) {
            EXPECT_EQ(contexts[i].decode_ip, expected_contexts[tp_rank][i].decode_ip);
            EXPECT_EQ(contexts[i].decode_port, expected_contexts[tp_rank][i].decode_port);
            EXPECT_EQ(contexts[i].local_partition_count, expected_contexts[tp_rank][i].local_partition_count);
            EXPECT_EQ(contexts[i].local_partition_id, expected_contexts[tp_rank][i].local_partition_id);
            EXPECT_EQ(contexts[i].remote_partition_count, expected_contexts[tp_rank][i].remote_partition_count);
            EXPECT_EQ(contexts[i].remote_partition_id, expected_contexts[tp_rank][i].remote_partition_id);
        }
    }
}

TEST_F(AsymmetricTpUtilTest, HandleND1P_InvalidDivisibility) {
    gpt_init_parameter_.tp_size_ = 2;
    gpt_init_parameter_.tp_rank_ = 0;

    AsymmetricTpUtil util(gpt_init_parameter_);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {
        {"192.168.1.10", 8080},
        {"192.168.1.11", 8080},
        {"192.168.1.12", 8080},
    };

    // decode_transfer_servers.size()=3 不能被 tp_size=2 整除，应该返回空
    auto contexts = util.handleAsymmetricTP(decode_transfer_servers);
    EXPECT_TRUE(contexts.empty());
}

}  // namespace rtp_llm
