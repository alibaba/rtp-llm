#pragma once

#include "navi/engine/Resource.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Data.h"
#include "navi/resource/GigClientR.h"
#include <grpc++/impl/codegen/byte_buffer.h>
#include <grpc++/impl/codegen/status.h>

namespace navi_sched {

// Completion owns the reply and admission token. The CQ thread only publishes
// this object; protobuf parsing and scheduling remain in Navi compute.
class TairKvcmGrpcCompletionData : public navi::Data {
public:
    TairKvcmGrpcCompletionData();
    size_t queryIndex = 0;
    uint64_t round = 0;
    bool discovery = false;
    bool eventOk = false;
    grpc::Status status;
    grpc::ByteBuffer response;
    int64_t startedUs = 0;
    int64_t completedUs = 0;
    int64_t deadlineUs = 0;
    size_t requestBytes = 0;
    size_t maxResponseBytes = 0;
    std::shared_ptr<void> queryToken;
    multi_call::ReplyPtr reply;
    // Called on Navi compute, never on a GIG completion thread.
    void unpack();
private:
    bool _unpacked = false;
};

class TairKvcmGrpcClientR : public navi::Resource {
public:
    struct Discovery : public autil::legacy::Jsonizable {
        void Jsonize(JsonWrapper &json) override {
            json.Jsonize("type", type, type);
            json.Jsonize("hosts", hosts, hosts);
            json.Jsonize("jmenv_dom", jmenvDom, jmenvDom);
        }
        bool isSpectrum() const { return type == "spectrum" || type == "dashscope"; }
        std::string type = "spectrum";
        std::vector<std::string> hosts;
        std::string jmenvDom;
    };
    struct Topology : public autil::legacy::Jsonizable {
        void Jsonize(JsonWrapper &json) override {
            json.Jsonize("address", address, address);
            json.Jsonize("port", port, port);
            json.Jsonize("discovery", discovery, discovery);
            json.Jsonize("namespace", instanceNamespace, instanceNamespace);
            json.Jsonize("thread_num", threadNum, threadNum);
        }
        bool valid() const;
        bool configured() const { return !address.empty(); }
        std::string address;
        std::string instanceNamespace;
        int32_t port = 6381;
        size_t threadNum = 4;
        Discovery discovery;
    };
    static const std::string RESOURCE_ID;
    TairKvcmGrpcClientR() = default;
    ~TairKvcmGrpcClientR() override = default;
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    const Topology &topology() const { return _topology; }
    // Reads an existing GIG snapshot; never performs discovery IO on compute.
    std::vector<std::string> bootstrapTargets() const;
    static constexpr int32_t MAX_TRANSPORT_BYTES = 64 * 1024 * 1024;
    using MessagePtr = std::shared_ptr<google::protobuf::Message>;
    // The immutable typed payload is serialized once by GIG unary.
    // False means no RPC was started and no callback will be delivered.
    bool call(const std::string &target, const std::string &method,
              const MessagePtr &payload, int64_t deadlineUs,
              size_t maxMessageBytes, const navi::AsyncPipePtr &pipe,
              size_t queryIndex, uint64_t round, bool discovery,
              const std::shared_ptr<void> &queryToken = {});
private:
    bool configureTransport();
    bool makeSubscription(const multi_call::MultiCallConfig &parent,
                          multi_call::SubscribeConfig &subscription) const;
    bool subscribe();
    static bool validTarget(const std::string &target);
    Topology _topology;
    bool _subscriptionReady = false;
    bool _transportReady = false;
    navi::GigClientR *_gigClientR = nullptr;
};

} // namespace navi_sched
