#pragma once

#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace autil {
class LockFreeThreadPool;
}

namespace http_server {
class HttpServer;
}

namespace rtp_llm {

class GangServer {
public:
    GangServer(py::object gang_info);
    virtual ~GangServer();

public:
    // virtual for test
    virtual void requestWorkers(const std::map<std::string, std::string>& body_map,
                                const std::string&                        uri     = "inference_internal",
                                bool                                      is_wait = false);

private:
    void getWorkers();

private:
    py::object                                 gang_info_;
    std::vector<std::pair<std::string, int>>   workers_;
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
};

}  // namespace rtp_llm
