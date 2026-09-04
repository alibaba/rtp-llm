#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "autil/NetUtil.h"
#include "http_client/SimpleHttpClient.h"
#include "http_server/HttpServer.h"
#include "rtp_llm/cpp/api_server/ConstraintTreeService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/models/logits_processor/ConstraintTreeCsr.h"

using namespace ::testing;

namespace rtp_llm {

namespace {

void appendU32(std::string& output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void appendU64(std::string& output, uint64_t value) {
    appendU32(output, static_cast<uint32_t>(value));
    appendU32(output, static_cast<uint32_t>(value >> 32));
}

std::string makeArtifact(uint64_t             version,
                         std::vector<int32_t> row_ptr        = {0, 1, 2},
                         std::vector<int32_t> col_idx        = {10, 2},
                         std::vector<int32_t> next_state     = {1, -1},
                         uint64_t             sid_count      = 1,
                         int32_t              start_token_id = 225,
                         int32_t              end_token_id   = 2) {
    std::string output("RTPCSR01", 8);
    appendU32(output, 1);
    appendU32(output, 48);
    appendU64(output, version);
    appendU32(output, static_cast<uint32_t>(start_token_id));
    appendU32(output, static_cast<uint32_t>(end_token_id));
    appendU32(output, static_cast<uint32_t>(row_ptr.size() - 1));
    appendU32(output, static_cast<uint32_t>(col_idx.size()));
    appendU64(output, sid_count);
    for (int32_t value : row_ptr) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    for (int32_t value : col_idx) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    for (int32_t value : next_state) {
        appendU32(output, static_cast<uint32_t>(value));
    }
    return output;
}

}  // namespace

class ConstraintTreeServiceTest: public ::testing::Test {
protected:
    struct Response {
        int         status_code;
        std::string body;
    };

    void SetUp() override {
        service_ = std::make_shared<ConstraintTreeService>();
    }

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    createHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* value) { value->free(); });
    }

    Response sendUpdate(const std::string& body) {
        auto        mock_writer = std::make_unique<http_server::MockHttpResponseWriter>();
        std::string response_body;
        EXPECT_CALL(*mock_writer, Write).WillOnce(Invoke([&](const std::string& data) {
            response_body = data;
            return true;
        }));
        std::unique_ptr<http_server::HttpResponseWriter> writer = std::move(mock_writer);
        http_server::HttpRequest                         request;
        request._request = createHttpPacket(body);

        service_->updateConstraintTree(writer, request);
        return {writer->_statusCode, std::move(response_body)};
    }

    Response readStatus() {
        auto        mock_writer = std::make_unique<http_server::MockHttpResponseWriter>();
        std::string response_body;
        EXPECT_CALL(*mock_writer, Write).WillOnce(Invoke([&](const std::string& data) {
            response_body = data;
            return true;
        }));
        std::unique_ptr<http_server::HttpResponseWriter> writer = std::move(mock_writer);
        http_server::HttpRequest                         request;

        service_->constraintTreeStatus(writer, request);
        return {writer->_statusCode, std::move(response_body)};
    }

    bool waitForState(const std::string& expected) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(service_->mutex_);
                if (service_->update_state_ == expected) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }

    std::shared_ptr<ConstraintTreeService> service_;
};

TEST_F(ConstraintTreeServiceTest, UpdateAndReadStatus) {
    const uint64_t    version = ConstraintTreeCsrManager::instance()->currentVersion() + 1;
    const std::string body    = makeArtifact(version);

    auto        mock_writer = std::make_unique<http_server::MockHttpResponseWriter>();
    std::string response_body;
    EXPECT_CALL(*mock_writer, Write).WillOnce(Invoke([&](const std::string& data) {
        response_body = data;
        return true;
    }));
    auto* raw_writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer.get());
    ASSERT_NE(nullptr, raw_writer);
    std::unique_ptr<http_server::HttpResponseWriter> writer(raw_writer);
    http_server::HttpRequest                         request;
    request._request = createHttpPacket(body);

    service_->updateConstraintTree(writer, request);
    EXPECT_EQ(200, writer->_statusCode);
    EXPECT_THAT(response_body, HasSubstr("\"status\":\"accepted\""));
    EXPECT_THAT(response_body, HasSubstr("\"requested_version\":" + std::to_string(version)));
    writer.release();

    ASSERT_TRUE(waitForState("ready"));
    ASSERT_EQ(version, ConstraintTreeCsrManager::instance()->currentVersion());

    auto status_writer = std::make_unique<http_server::MockHttpResponseWriter>();
    EXPECT_CALL(*status_writer, Write).WillOnce(Invoke([&](const std::string& data) {
        response_body = data;
        return true;
    }));
    auto* status_raw_writer = dynamic_cast<http_server::HttpResponseWriter*>(status_writer.get());
    ASSERT_NE(nullptr, status_raw_writer);
    std::unique_ptr<http_server::HttpResponseWriter> status_writer_ptr(status_raw_writer);
    http_server::HttpRequest                         status_request;
    service_->constraintTreeStatus(status_writer_ptr, status_request);
    EXPECT_THAT(response_body, HasSubstr("\"status\":\"ready\""));
    EXPECT_THAT(response_body, HasSubstr("\"prefix_count\":2"));
    EXPECT_THAT(response_body, HasSubstr("\"edge_count\":2"));
    status_writer_ptr.release();
}

TEST_F(ConstraintTreeServiceTest, RejectsStaleVersion) {
    auto           manager        = ConstraintTreeCsrManager::instance();
    const uint64_t active_version = std::max<uint64_t>(2, manager->currentVersion() + 1);
    ASSERT_TRUE(manager->updateFromBinary(makeArtifact(active_version), nullptr).ok());

    const std::string body        = makeArtifact(active_version - 1, {0, 1, 2}, {11, 2}, {1, -1});
    auto              mock_writer = std::make_unique<http_server::MockHttpResponseWriter>();
    EXPECT_CALL(*mock_writer, Write).WillOnce(Return(true));
    auto* raw_writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer.get());
    ASSERT_NE(nullptr, raw_writer);
    std::unique_ptr<http_server::HttpResponseWriter> writer(raw_writer);
    http_server::HttpRequest                         request;
    request._request = createHttpPacket(body);

    service_->updateConstraintTree(writer, request);
    EXPECT_EQ(409, writer->_statusCode);
    EXPECT_EQ(active_version, manager->currentVersion());
    writer.release();
}

TEST_F(ConstraintTreeServiceTest, RejectsInvalidBinaryWithoutDroppingCurrentTree) {
    auto           manager        = ConstraintTreeCsrManager::instance();
    const uint64_t active_version = manager->currentVersion();
    const auto     active_tree    = manager->snapshot();

    auto mock_writer = std::make_unique<http_server::MockHttpResponseWriter>();
    EXPECT_CALL(*mock_writer, Write).WillOnce(Return(true));
    auto* raw_writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer.get());
    ASSERT_NE(nullptr, raw_writer);
    std::unique_ptr<http_server::HttpResponseWriter> writer(raw_writer);
    http_server::HttpRequest                         request;
    request._request = createHttpPacket("not-json");

    service_->updateConstraintTree(writer, request);
    EXPECT_EQ(400, writer->_statusCode);
    EXPECT_EQ(active_version, manager->currentVersion());
    EXPECT_EQ(active_tree, manager->snapshot());
    writer.release();
}

TEST_F(ConstraintTreeServiceTest, AtomicSwapKeepsPinnedSnapshotAlive) {
    auto           manager       = ConstraintTreeCsrManager::instance();
    const uint64_t first_version = manager->currentVersion() + 1;
    ASSERT_EQ(ConstraintTreeCsrUpdateCode::UPDATED,
              manager->updateFromBinary(makeArtifact(first_version, {0, 1, 2}, {10, 2}, {1, -1}), nullptr).code);
    const auto pinned = manager->snapshot();
    ASSERT_NE(nullptr, pinned);
    ASSERT_EQ(1, pinned->transition(0, 10));

    const uint64_t second_version = first_version + 1;
    ASSERT_EQ(ConstraintTreeCsrUpdateCode::UPDATED,
              manager->updateFromBinary(makeArtifact(second_version, {0, 1, 2}, {11, 2}, {1, -1}), nullptr).code);
    const auto active = manager->snapshot();

    ASSERT_NE(nullptr, active);
    EXPECT_EQ(second_version, active->version());
    EXPECT_EQ(1, active->transition(0, 11));
    EXPECT_EQ(ConstraintTreeCsrSnapshot::INVALID_TRANSITION, active->transition(0, 10));
    EXPECT_EQ(first_version, pinned->version());
    EXPECT_EQ(1, pinned->transition(0, 10));
    EXPECT_EQ(ConstraintTreeCsrSnapshot::INVALID_TRANSITION, pinned->transition(0, 11));
}

TEST_F(ConstraintTreeServiceTest, InvalidBackgroundLoadKeepsCurrentTreeAndSameVersionCanRetry) {
    auto manager = ConstraintTreeCsrManager::instance();
    if (!manager->snapshot()) {
        ASSERT_TRUE(manager->updateFromBinary(makeArtifact(1), nullptr).ok());
    }
    const uint64_t failed_version = manager->currentVersion() + 1;
    const auto     active_tree    = manager->snapshot();

    const auto accepted = sendUpdate(makeArtifact(failed_version, {0, 1}, {10}, {-2}));
    ASSERT_EQ(200, accepted.status_code);
    EXPECT_THAT(accepted.body, HasSubstr("\"status\":\"accepted\""));
    ASSERT_TRUE(waitForState("failed"));
    EXPECT_EQ(active_tree, manager->snapshot());

    const auto failed_status = readStatus();
    EXPECT_EQ(200, failed_status.status_code);
    EXPECT_THAT(failed_status.body, HasSubstr("\"status\":\"failed\""));
    EXPECT_THAT(failed_status.body, HasSubstr("\"version\":" + std::to_string(active_tree->version())));

    const auto retry = sendUpdate(makeArtifact(failed_version, {0, 1, 2}, {12, 2}, {1, -1}));
    ASSERT_EQ(200, retry.status_code);
    EXPECT_THAT(retry.body, HasSubstr("\"status\":\"accepted\""));
    ASSERT_TRUE(waitForState("ready"));
    EXPECT_EQ(failed_version, manager->currentVersion());

    const auto duplicate = sendUpdate(makeArtifact(failed_version, {0, 1, 2}, {99, 2}, {1, -1}));
    EXPECT_EQ(200, duplicate.status_code);
    EXPECT_THAT(duplicate.body, HasSubstr("\"status\":\"already_current\""));
    EXPECT_EQ(12, manager->snapshot()->colIdx().front());
}

TEST_F(ConstraintTreeServiceTest, RealHttpEndpointsQueueActivateAndReportStatus) {
    const auto        port    = autil::NetUtil::randomPort();
    const std::string address = "tcp:127.0.0.1:" + std::to_string(port);
    auto              server  = std::make_shared<http_server::HttpServer>();
    ASSERT_TRUE(server->RegisterRoute("POST",
                                      "/update_constraint_tree",
                                      [service = service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                           const http_server::HttpRequest&                  request) {
                                          service->updateConstraintTree(writer, request);
                                      }));
    ASSERT_TRUE(server->RegisterRoute("GET",
                                      "/constraint_tree_status",
                                      [service = service_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                           const http_server::HttpRequest&                  request) {
                                          service->constraintTreeStatus(writer, request);
                                      }));
    ASSERT_TRUE(server->Start(address));

    auto              client  = std::make_shared<http_server::SimpleHttpClient>();
    const uint64_t    version = ConstraintTreeCsrManager::instance()->currentVersion() + 1;
    const std::string body = makeArtifact(version, {0, 1, 2, 3}, {169967, 216546, 151645}, {1, 2, -1}, 1, 1699, 151645);
    std::promise<std::pair<bool, std::string>> update_promise;
    auto                                       update_future = update_promise.get_future();
    ASSERT_TRUE(
        client->post(address, "/update_constraint_tree", body, [&update_promise](bool ok, const std::string& data) {
            update_promise.set_value({ok, data});
        }));
    ASSERT_EQ(std::future_status::ready, update_future.wait_for(std::chrono::seconds(5)));
    const auto update_response = update_future.get();
    ASSERT_TRUE(update_response.first);
    EXPECT_THAT(update_response.second, HasSubstr("\"status\":\"accepted\""));

    ASSERT_TRUE(waitForState("ready"));
    ASSERT_EQ(version, ConstraintTreeCsrManager::instance()->currentVersion());

    std::promise<std::pair<bool, std::string>> status_promise;
    auto                                       status_future = status_promise.get_future();
    ASSERT_TRUE(
        client->get(address, "/constraint_tree_status", "", [&status_promise](bool ok, const std::string& data) {
            status_promise.set_value({ok, data});
        }));
    ASSERT_EQ(std::future_status::ready, status_future.wait_for(std::chrono::seconds(5)));
    const auto status_response = status_future.get();
    ASSERT_TRUE(status_response.first);
    EXPECT_THAT(status_response.second, HasSubstr("\"status\":\"ready\""));
    EXPECT_THAT(status_response.second, HasSubstr("\"version\":" + std::to_string(version)));
    EXPECT_THAT(status_response.second, HasSubstr("\"prefix_count\":3"));
    server->Stop();
}

}  // namespace rtp_llm
