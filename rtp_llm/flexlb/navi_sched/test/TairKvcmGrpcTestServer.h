#pragma once
#include <grpc++/grpc++.h>
#include <grpc++/generic/async_generic_service.h>
#include <functional>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>
namespace navi_sched { namespace test {
std::string bytes(const grpc::ByteBuffer &buffer)
{
    std::vector<grpc::Slice> slices;
    if (!buffer.Dump(&slices).ok()) return {};
    std::string result;
    for (const auto &slice : slices)
        result.append(reinterpret_cast<const char *>(slice.begin()), slice.size());
    return result;
}

class KvcmGrpcMock {
    struct Call {
        enum State { ACCEPT, READ, WRITE, FINISH } state = ACCEPT;
        explicit Call(KvcmGrpcMock &server) : owner(server), stream(&context) {
            owner.service.RequestCall(&context, &stream, owner.cq.get(), owner.cq.get(), this);
        }
        KvcmGrpcMock &owner;
        grpc::GenericServerContext context;
        grpc::GenericServerAsyncReaderWriter stream;
        grpc::ByteBuffer request, response;
        void reply(const std::string &payload) {
            grpc::Slice slice(payload.data(), payload.size());
            response = grpc::ByteBuffer(&slice, 1);
            state = WRITE;
            stream.Write(response, this);
        }
        void advance(bool ok) {
            std::lock_guard<std::mutex> operation(owner.operationMutex);
            if (owner.stopping || !ok) { delete this; return; }
            if (state == ACCEPT) {
                if (!ok) { delete this; return; }
                new Call(owner);
                state = READ;
                stream.Read(&request, this);
            } else if (state == READ) {
                if (!ok) {
                    state = FINISH;
                    stream.Finish(grpc::Status(grpc::StatusCode::CANCELLED, "read cancelled"), this);
                    return;
                }
                const auto payload = bytes(request);
                {
                    std::lock_guard<std::mutex> lock(owner.mutex);
                    owner.requests.emplace_back(context.method(), payload);
                    owner.condition.notify_all();
                }
                const auto result = owner.handler(context.method(), payload);
                if (result.hold) {
                    std::lock_guard<std::mutex> lock(owner.mutex);
                    owner.held.push_back(this);
                } else if (!result.status.ok()) {
                    state = FINISH;
                    stream.Finish(result.status, this);
                } else reply(result.payload);
            } else if (state == WRITE) {
                state = FINISH;
                stream.Finish(grpc::Status::OK, this);
            } else delete this;
        }
    };
public:
    struct Reply { std::string payload; grpc::Status status; bool hold = false; };
    using Handler = std::function<Reply(const std::string &, const std::string &)>;
    explicit KvcmGrpcMock(Handler handler, std::string bindIp = "127.0.0.1") : handler(std::move(handler)), bindIp(std::move(bindIp)) {
        if (this->bindIp != "127.0.0.1" && this->bindIp != "127.0.0.2") throw std::runtime_error("loopback only");
        grpc::ServerBuilder builder;
        builder.AddListeningPort(this->bindIp + ":0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterAsyncGenericService(&service);
        cq = builder.AddCompletionQueue();
        server = builder.BuildAndStart();
        if (!server || port <= 0) throw std::runtime_error("loopback gRPC server failed");
        new Call(*this);
        worker = std::thread([this] {
            void *tag = nullptr; bool ok = false;
            while (cq->Next(&tag, &ok)) static_cast<Call *>(tag)->advance(ok);
        });
    }
    ~KvcmGrpcMock() {
        release();
        {
            std::lock_guard<std::mutex> operation(operationMutex);
            stopping = true;
        }
        server->Shutdown(std::chrono::system_clock::now());
        cq->Shutdown();
        worker.join();
    }
    std::string target() const { return bindIp + ":" + std::to_string(port); }
    bool waitRequests(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::seconds(3), [&] { return requests.size() >= count; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto call : held) call->reply("released");
        held.clear();
    }
    Handler handler;
    std::string bindIp;
    int port = 0;
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::pair<std::string, std::string>> requests;
private:
    std::mutex operationMutex;
    bool stopping = false;
    grpc::AsyncGenericService service;
    std::unique_ptr<grpc::ServerCompletionQueue> cq;
    std::unique_ptr<grpc::Server> server;
    std::thread worker;
    std::vector<Call *> held;
};
} }
