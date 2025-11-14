#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/AtomicCounter.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "rtp_llm/cpp/api_server/GangServer.h"
#include "rtp_llm/cpp/api_server/LoraService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/test/mock/MockApiServerMetricReporter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockGangServer.h"
#include "rtp_llm/cpp/api_server/test/mock/MockLoraManager.h"
#include "rtp_llm/cpp/api_server/test/mock/MockWeightsLoader.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"

using namespace ::testing;
namespace rtp_llm {

class LoraServiceTest: public ::testing::Test {
public:
    LoraServiceTest()           = default;
    ~LoraServiceTest() override = default;

protected:
    void SetUp() override {
        mock_engine_ = std::make_shared<MockEngineBase>();
        auto engine  = std::dynamic_pointer_cast<EngineBase>(mock_engine_);

        mock_lora_manager_    = std::make_shared<lora::MockLoraManager>();
        auto lora_manager     = std::dynamic_pointer_cast<lora::LoraManager>(mock_lora_manager_);
        engine->lora_manager_ = lora_manager;

        mock_metric_reporter_ = std::make_shared<MockApiServerMetricReporter>();
        auto metric_reporter  = std::dynamic_pointer_cast<ApiServerMetricReporter>(mock_metric_reporter_);

        mock_gang_server_ = std::make_shared<MockGangServer>();
        auto gang_server  = std::dynamic_pointer_cast<GangServer>(mock_gang_server_);

        mock_weights_loader_ = std::make_shared<MockWeightsLoader>();
        auto weights_loader  = std::dynamic_pointer_cast<WeightsLoader>(mock_weights_loader_);

        ON_CALL(*mock_weights_loader_, loadLoraWeights(::testing::_, ::testing::_))
            .WillByDefault(::testing::Invoke([](const std::string&, const std::string&) {
                return std::make_pair(std::make_unique<rtp_llm::lora::loraLayerWeightsMap>(),
                                      std::make_unique<rtp_llm::lora::loraLayerWeightsMap>());
            }));

        std::map<std::string, std::string> lora_infos;
        lora_service_ = std::make_shared<LoraService>(engine, gang_server, weights_loader, lora_infos, metric_reporter);
        mock_writer_  = std::make_unique<http_server::MockHttpResponseWriter>();
    }
    void TearDown() override {
        lora_service_.reset();
    }

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }
    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::map<std::string, std::string>& body_map) {
        auto packet        = new ::anet::HTTPPacket();
        auto body_json_str = ::autil::legacy::ToJsonString(body_map);
        packet->setBody(body_json_str.c_str(), body_json_str.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }

    void SetToMaster() {
        auto& parallel_info = ParallelInfo::globalParallelInfo();
        parallel_info.setTpSize(1);
        parallel_info.setPpSize(2);
        parallel_info.setWorldRank(0);
        parallel_info.setWorldSize(2);
        parallel_info.setLocalWorldSize(1);
        parallel_info.setDpSize(1);
        parallel_info.setEpSize(1);
    }
    void SetToWorker() {
        auto& parallel_info = ParallelInfo::globalParallelInfo();
        parallel_info.setTpSize(1);
        parallel_info.setPpSize(2);
        parallel_info.setWorldRank(1);
        parallel_info.setWorldSize(2);
        parallel_info.setLocalWorldSize(1);
        parallel_info.setDpSize(1);
        parallel_info.setEpSize(1);
    }

protected:
    std::shared_ptr<MockEngineBase>                      mock_engine_;
    std::shared_ptr<lora::MockLoraManager>               mock_lora_manager_;
    std::shared_ptr<MockApiServerMetricReporter>         mock_metric_reporter_;
    std::shared_ptr<MockGangServer>                      mock_gang_server_;
    std::shared_ptr<MockWeightsLoader>                   mock_weights_loader_;
    std::shared_ptr<LoraService>                         lora_service_;
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
};

TEST_F(LoraServiceTest, Constructor) {
    {
        std::map<std::string, std::string> lora_infos;
        LoraService                        lora_service(nullptr, nullptr, nullptr, lora_infos, nullptr);
        EXPECT_EQ(lora_service.engine_, nullptr);
        EXPECT_EQ(lora_service.gang_server_, nullptr);
        EXPECT_EQ(lora_service.metric_reporter_, nullptr);
    }
    {
        auto engine          = std::dynamic_pointer_cast<EngineBase>(mock_engine_);
        auto gang_server     = std::dynamic_pointer_cast<GangServer>(mock_gang_server_);
        auto weights_loader  = std::dynamic_pointer_cast<WeightsLoader>(mock_weights_loader_);
        auto metric_reporter = std::dynamic_pointer_cast<ApiServerMetricReporter>(mock_metric_reporter_);
        std::map<std::string, std::string> lora_infos;
        LoraService lora_service(engine, gang_server, weights_loader, lora_infos, metric_reporter);
        EXPECT_NE(lora_service.engine_, nullptr);
        EXPECT_NE(lora_service.metric_reporter_, nullptr);
        EXPECT_NE(lora_service.gang_server_, nullptr);
    }
}

TEST_F(LoraServiceTest, Update_IsNotMaster) {
    // 模拟不是 master 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    try {
        lora_service_->update(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNSUPPORTED_OPERATION);
        EXPECT_EQ(e.getMessage(), "gang worker should not access /update api directly");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION instead of std::exception";
    }

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, Update_EngineIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    lora_service_->engine_ = nullptr;
    try {
        lora_service_->update(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "update failed, engine is null");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, Update_LoraManagerIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    lora_service_->engine_->lora_manager_ = nullptr;
    try {
        lora_service_->update(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "update failed, lora manager is null");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, Update_ParseRequestBodyFailed) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // request body 为空, 模拟解析 request body 失败的情况
    http_server::HttpRequest request;
    EXPECT_THROW(lora_service_->update(writer_ptr, request), autil::legacy::ExceptionBase);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, Update_Success) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del({
    "peft_info": {
        "lora_info": {
            "test0": "/path/to/lora/test/",
            "test1": "/path/to/lora/test/"
        }
    }
})del";
    request._request              = CreateHttpPacket(body);

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"("null")");
        return true;
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportUpdateQpsMetric).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportUpdateLatencyMs).WillOnce(Invoke([](double val) {
        EXPECT_TRUE(val >= 0);
    }));

    // 模拟 lora manager 中的 lora map
    lora_service_->lora_info_map_ = {{"test0", "/path0"}, {"test1", "/path/to/lora/test/"}, {"test2", "/path2"}};
    // 只有 test0 和 test2 会被 remove, test1 不会被 remove
    EXPECT_CALL(*mock_lora_manager_, removeLora("test0")).WillOnce(Return());
    EXPECT_CALL(*mock_lora_manager_, removeLora("test2")).WillOnce(Return());
    EXPECT_CALL(*mock_lora_manager_, removeLora("test1")).Times(0);

    // test0 会被 add, test1 由于已经在 lora manager 中存在所以不会被 add
    EXPECT_CALL(*mock_lora_manager_, addLora("test0", _, _)).WillOnce(Return());
    EXPECT_CALL(*mock_lora_manager_, addLora("test1", _, _)).Times(0);

    lora_service_->update(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraInternal_IsNotWorker) {
    // 模拟不是 worker 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    try {
        lora_service_->addLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNSUPPORTED_OPERATION);
        EXPECT_EQ(e.getMessage(), "gang master should not access /add_lora_internal api directly");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraInternal_EngineIsNull) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    lora_service_->engine_ = nullptr;
    try {
        lora_service_->addLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "add lora internal failed, engine is null");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraInternal_ParseRequestBodyFailed) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // request body 为空, 模拟解析 request body 失败的情况
    http_server::HttpRequest request;
    EXPECT_THROW(lora_service_->addLoraInternal(writer_ptr, request), autil::legacy::ExceptionBase);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraInternal_RequestBodyHasNoLoraPath) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest           request;
    std::map<std::string, std::string> body_map = {{"adapter_name", "test"}};
    request._request                            = CreateHttpPacket(body_map);

    try {
        lora_service_->addLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "add lora internal failed, request has no adapter_name or lora_path");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraInternal_Success) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest           request;
    const std::string                  adapter_name = "test_adapter_name";
    const std::string                  lora_path    = "/test/lora_path";
    std::map<std::string, std::string> body_map     = {{"adapter_name", adapter_name}, {"lora_path", lora_path}};
    request._request                                = CreateHttpPacket(body_map);

    EXPECT_CALL(*mock_lora_manager_, addLora(adapter_name, _, _)).Times(1);
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"("null")");
        return true;
    }));

    lora_service_->addLoraInternal(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, RemoveLoraInternal_IsNotWorker) {
    // 模拟不是 worker 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    try {
        lora_service_->removeLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNSUPPORTED_OPERATION);
        EXPECT_EQ(e.getMessage(), "gang master should not access /remove_lora_internal api directly");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNSUPPORTED_OPERATION instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, RemoveLoraInternal_EngineIsNull) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    lora_service_->engine_ = nullptr;
    try {
        lora_service_->removeLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "remove lora internal failed, engine is null");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, RemoveLoraInternal_ParseRequestBodyFailed) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // request body 为空, 模拟解析 request body 失败的情况
    http_server::HttpRequest request;
    EXPECT_THROW(lora_service_->removeLoraInternal(writer_ptr, request), autil::legacy::ExceptionBase);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, RemoveLoraInternal_RequestBodyHasNoAdapterName) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest           request;
    std::map<std::string, std::string> body_map = {{"no_adapter_name", "test"}};
    request._request                            = CreateHttpPacket(body_map);

    try {
        lora_service_->removeLoraInternal(writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "remove lora internal failed, request has no adapter_name or lora_path");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }
    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, RemoveLoraInternal_Success) {
    // 模拟是 worker 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest           request;
    const std::string                  adapter_name = "test_adapter_name";
    const std::string                  lora_path    = "/test/lora_path";
    std::map<std::string, std::string> body_map     = {{"adapter_name", adapter_name}, {"lora_path", lora_path}};
    request._request                                = CreateHttpPacket(body_map);

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"("null")");
        return true;
    }));
    EXPECT_CALL(*mock_lora_manager_, removeLora(adapter_name));

    lora_service_->removeLoraInternal(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(LoraServiceTest, AddLoraFailed_GangServerIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    lora_service_->gang_server_ = nullptr;
    EXPECT_FALSE(lora_service_->addLora("test", "/test"));
}

TEST_F(LoraServiceTest, AddLoraFailed_EngineIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    const std::string adapter_name = "test_adapter_name";
    const std::string lora_path    = "/test/lora_path";
    EXPECT_CALL(*mock_gang_server_, requestWorkers(_, "add_lora_internal", true))
        .WillOnce(Invoke([adapter_name, lora_path](
                             const std::map<std::string, std::string>& body_map, const std::string& uri, bool is_wait) {
            EXPECT_EQ(body_map.size(), 2);
            EXPECT_TRUE(body_map.count("adapter_name") > 0);
            EXPECT_EQ(body_map.at("adapter_name"), adapter_name);
            EXPECT_TRUE(body_map.count("lora_path") > 0);
            EXPECT_EQ(body_map.at("lora_path"), lora_path);
        }));

    lora_service_->engine_ = nullptr;
    EXPECT_FALSE(lora_service_->addLora(adapter_name, lora_path));
}

TEST_F(LoraServiceTest, AddLoraFailed_LoraManagerIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    const std::string adapter_name = "test_adapter_name";
    const std::string lora_path    = "/test/lora_path";
    EXPECT_CALL(*mock_gang_server_, requestWorkers(_, "add_lora_internal", true))
        .WillOnce(Invoke([adapter_name, lora_path](
                             const std::map<std::string, std::string>& body_map, const std::string& uri, bool is_wait) {
            EXPECT_EQ(body_map.size(), 2);
            EXPECT_TRUE(body_map.count("adapter_name") > 0);
            EXPECT_EQ(body_map.at("adapter_name"), adapter_name);
            EXPECT_TRUE(body_map.count("lora_path") > 0);
            EXPECT_EQ(body_map.at("lora_path"), lora_path);
        }));

    lora_service_->engine_->lora_manager_ = nullptr;
    EXPECT_FALSE(lora_service_->addLora(adapter_name, lora_path));
}

TEST_F(LoraServiceTest, AddLoraSuccess) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    const std::string adapter_name = "test_adapter_name";
    const std::string lora_path    = "/test/lora_path";
    EXPECT_CALL(*mock_gang_server_, requestWorkers(_, "add_lora_internal", true))
        .WillOnce(Invoke([adapter_name, lora_path](
                             const std::map<std::string, std::string>& body_map, const std::string& uri, bool is_wait) {
            EXPECT_EQ(body_map.size(), 2);
            EXPECT_TRUE(body_map.count("adapter_name") > 0);
            EXPECT_EQ(body_map.at("adapter_name"), adapter_name);
            EXPECT_TRUE(body_map.count("lora_path") > 0);
            EXPECT_EQ(body_map.at("lora_path"), lora_path);
        }));
    EXPECT_CALL(*mock_lora_manager_, addLora(adapter_name, _, _)).Times(1);
    EXPECT_TRUE(lora_service_->addLora(adapter_name, lora_path));
}

TEST_F(LoraServiceTest, RemoveLoraFailed_EngineIsNull) {
    lora_service_->engine_         = nullptr;
    const std::string adapter_name = "test_adapter_name";
    EXPECT_FALSE(lora_service_->removeLora(adapter_name));
}

TEST_F(LoraServiceTest, RemoveLoraFailed_LoraManagerIsNull) {
    lora_service_->engine_->lora_manager_ = nullptr;
    const std::string adapter_name        = "test_adapter_name";
    EXPECT_FALSE(lora_service_->removeLora(adapter_name));
}

TEST_F(LoraServiceTest, RemoveLoraFailed_GangServerIsNull) {
    const std::string adapter_name = "test_adapter_name";
    EXPECT_CALL(*mock_lora_manager_, removeLora(adapter_name));

    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    lora_service_->gang_server_ = nullptr;
    EXPECT_FALSE(lora_service_->removeLora(adapter_name));
}

TEST_F(LoraServiceTest, RemoveLoraSuccess) {
    const std::string adapter_name = "test_adapter_name";
    EXPECT_CALL(*mock_lora_manager_, removeLora(adapter_name));

    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(ParallelInfo::globalParallelInfo().isMaster());
    EXPECT_TRUE(parallel_info.getWorldSize() > 1);

    EXPECT_CALL(*mock_gang_server_, requestWorkers(_, "remove_lora_internal", true))
        .WillOnce(Invoke(
            [adapter_name](const std::map<std::string, std::string>& body_map, const std::string& uri, bool is_wait) {
                EXPECT_EQ(body_map.size(), 1);
                EXPECT_TRUE(body_map.count("adapter_name") > 0);
                EXPECT_EQ(body_map.at("adapter_name"), adapter_name);
            }));

    EXPECT_TRUE(lora_service_->removeLora(adapter_name));
}

}  // namespace rtp_llm
