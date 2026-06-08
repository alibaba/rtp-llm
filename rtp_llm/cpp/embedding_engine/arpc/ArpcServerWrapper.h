#pragma once

#include <memory>
#include <google/protobuf/service.h>

namespace rtp_llm {
class ArpcServerWrapper {
public:
    ArpcServerWrapper(
        std::unique_ptr<::google::protobuf::Service> service, int threadNum, int queueNum, int ioThreadNum, int port):
        service_(std::move(service)),
        port_(port),
        threadNum_(threadNum),
        queueNum_(queueNum),
        ioThreadNum_(ioThreadNum) {}
    virtual ~ArpcServerWrapper() = default;
    virtual void start()         = 0;
    virtual void stop()          = 0;

protected:
    std::unique_ptr<::google::protobuf::Service> service_;
    int                                          port_;
    int                                          threadNum_;
    int                                          queueNum_;
    int                                          ioThreadNum_;
};

}  // namespace rtp_llm
