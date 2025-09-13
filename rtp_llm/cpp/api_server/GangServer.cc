#include "rtp_llm/cpp/api_server/GangServer.h"

#include <numeric>
#include "autil/EnvUtil.h"
#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"
#include "autil/AtomicCounter.h"
#include "autil/LockFreeThreadPool.h"
#include "autil/WorkItem.h"

#include "http_server/HttpServer.h"
#include "http_client/SimpleHttpClient.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

class GangServerRequestWorkItem: public autil::WorkItem {
public:
    GangServerRequestWorkItem(const std::string&                               server_addr,
                              const std::string&                               route,
                              const std::string&                               body_json_str,
                              autil::AtomicCounter*                            counter,
                              std::shared_ptr<::http_server::SimpleHttpClient> http_client):
        server_addr_(server_addr),
        route_(route),
        body_json_str_(body_json_str),
        counter_(counter),
        http_client_(http_client) {}
    ~GangServerRequestWorkItem() override {
        counter_ = nullptr;
    }

public:
    void process() override {
        if (counter_) {
            counter_->dec();
        }
        if (!http_client_) {
            RTP_LLM_LOG_WARNING("process gang server request work item failed, http client is null");
            return;
        }
        auto succ = http_client_->post(server_addr_, route_, body_json_str_, nullptr);
        if (!succ) {
            RTP_LLM_LOG_WARNING("process gang server request work item failed, http client post failed, "
                                "server addr: %s, route: %s, request body: %s",
                                server_addr_.c_str(),
                                route_.c_str(),
                                body_json_str_.c_str());
        }
    }

private:
    std::string                                      server_addr_;
    std::string                                      route_;
    std::string                                      body_json_str_;
    autil::AtomicCounter*                            counter_{nullptr};
    std::shared_ptr<::http_server::SimpleHttpClient> http_client_;
};

GangServer::GangServer(py::object gang_info): gang_info_(gang_info) {
    const int thread_num = ParallelInfo::globalParallelInfo().getWorldSize();
    const int queue_size = 100;
    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(thread_num, queue_size, nullptr, "GangServerThreadPool");
    thread_pool_->start();
}

GangServer::~GangServer() {
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }
}

void GangServer::getWorkers() {
    if (workers_.empty()) {
        std::vector<std::pair<std::string, int>> workers_cpp;
        py::list                                 workers_py = gang_info_.attr("workers")();
        for (const auto& worker : workers_py) {
            std::string ip   = worker.attr("ip").cast<std::string>();
            int         port = worker.attr("server_port").cast<int>();
            workers_cpp.emplace_back(std::make_pair(ip, port));
        }
        workers_ = workers_cpp;
    }
}

void GangServer::requestWorkers(const std::map<std::string, std::string>& body_map,
                                const std::string&                        uri,
                                bool                                      is_wait) {
    getWorkers();
    autil::AtomicCounter counter;
    auto                 http_client = std::make_shared<::http_server::SimpleHttpClient>();
    for (const auto& worker : workers_) {
        counter.inc();
        std::string       ip            = worker.first;
        int               port          = worker.second;
        const std::string server_addr   = "tcp:" + ip + ":" + std::to_string(port);
        const std::string route         = "/" + uri;
        const std::string body_json_str = ::autil::legacy::ToJsonString(body_map, true);
        auto work_item = new GangServerRequestWorkItem(server_addr, route, body_json_str, &counter, http_client);
        auto code      = thread_pool_->pushWorkItem(work_item, false);
        if (code != autil::ThreadPool::ERROR_NONE) {
            RTP_LLM_LOG_WARNING(
                "gang server request worker failed, push work item failed, error code: %d, url: %s, body: %s",
                code,
                (server_addr + route).c_str(),
                body_json_str.c_str());
            work_item->destroy();
            counter.dec();
        }
    }
    if (is_wait) {
        while (counter.getValue() != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

}  // namespace rtp_llm
