#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/legacy/json.h"
#include "rtp_llm/cpp/api_server/ModelStatusService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;
namespace rtp_llm {

class ModelStatusServiceTest: public ::testing::Test {
protected:
    void SetUp() override {
        mock_writer_    = std::make_unique<http_server::MockHttpResponseWriter>();
        status_service_ = std::make_shared<ModelStatusService>();
    }
    void TearDown() override {}

protected:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<ModelStatusService>                  status_service_;
};

TEST_F(ModelStatusServiceTest, ModelStatus) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) -> bool {
        EXPECT_FALSE(data.empty());
        EXPECT_NO_THROW(autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::json::ParseJson(data)));
        auto json_map = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::json::ParseJson(data));
        EXPECT_TRUE(json_map.find("data") != json_map.end());
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    status_service_->modelStatus(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

}  // namespace rtp_llm