#include "aios/navi_sched/collector/PPCacheCollectGraph.h"
#include "navi/builder/GraphBuilder.h"
#include <chrono>
#include <set>
#include "autil/Log.h"

namespace navi_sched {
AUTIL_DECLARE_AND_SETUP_LOGGER(navi_sched, PPCacheCollectGraph);
namespace {
class ShardClosure : public navi::NaviUserResultClosure {
public:
    ShardClosure() { waitDestruct = true; }
    void run(navi::NaviUserResultPtr result,
             const std::shared_ptr<navi::ComputeScope> &) override
    {
        bool eof = false;
        while (result && !eof) {
            navi::NaviUserData data;
            result->nextData(data, eof);
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (result && result->getNaviResult() && result->getNaviResult()->hasError())
            _error = result->getNaviResult()->getErrorMessage();
        _completed = true;
        _condition.notify_all();
    }
    void wait()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [&]() { return _completed; });
    }
    std::string error()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _error;
    }

private:
    std::string _error;
    std::mutex _mutex;
    std::condition_variable _condition;
    bool _completed = false;
};
}

PPCacheCollectGraph::PPCacheCollectGraph(
    const navi::NaviPtr &navi, const std::string &localBiz, const std::string &group,
    const std::shared_ptr<PPCacheMergeR> &merge,
    const std::shared_ptr<PPRequestFinishedR> &requestFinished)
    : _navi(navi)
    , _localBiz(localBiz)
    , _group(group)
    , _merge(merge)
    , _requestFinished(requestFinished)
{
}
PPCacheCollectGraph::~PPCacheCollectGraph() { stop(); }

bool PPCacheCollectGraph::start(const std::vector<PPCacheShardEndpoint> &endpoints)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_navi || !_merge || !_requestFinished || _localBiz.empty() || _group.empty() ||
        endpoints.empty() || !_threads.empty())
        return false;
    std::set<uint32_t> ids;
    for (const auto &endpoint : endpoints) {
        if (endpoint.shardId >= endpoints.size() || endpoint.bizName.empty() ||
            endpoint.host.empty() || endpoint.grpcPort <= 0 ||
            endpoint.grpcPort > 65535 || !ids.insert(endpoint.shardId).second)
            return false;
    }
    _stopping = false;
    _active.resize(endpoints.size());
    for (const auto &endpoint : endpoints) {
        _threads.emplace_back([this, endpoint, count = uint32_t(endpoints.size())]() {
            collect(endpoint, count);
        });
    }
    return true;
}

std::unique_ptr<navi::GraphDef> PPCacheCollectGraph::build(
    const PPCacheShardEndpoint &endpoint, uint32_t shardCount, uint64_t generation) const
{
    auto graph = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graph.get());
    const auto remote = builder.newSubGraph(endpoint.bizName);
    builder.subGraph(remote).directLocation(endpoint.bizName, endpoint.host,
                                            endpoint.grpcPort);
    autil::legacy::json::JsonMap attributes;
    attributes["pp_group"] = _group;
    attributes["shard_id"] = endpoint.shardId;
    attributes["shard_count"] = shardCount;
    attributes["stream_generation"] = generation;
    attributes["collector_id"] = _localBiz;
    const auto source = builder.node("pp_cache_shard_source")
                            .kernel("navi_sched.pp_cache_shard_source_k")
                            .jsonAttrsFromJsonMap(attributes);
    const auto local = builder.newSubGraph(_localBiz);
    builder.subGraph(local).location(_localBiz, 1, 0);
    const auto sink = builder.node("pp_cache_merge")
                          .kernel("navi_sched.pp_cache_merge_k")
                          .jsonAttrsFromJsonMap(attributes);
    const auto finishedSource = builder.node("pp_request_finished_source")
                                    .kernel("navi_sched.pp_request_finished_source_k")
                                    .jsonAttrsFromJsonMap(attributes);
    builder.subGraph(remote);
    const auto finishedSink = builder.node("pp_request_finished_sink")
                                  .kernel("navi_sched.pp_request_finished_sink_k")
                                  .jsonAttrsFromJsonMap(attributes);
    finishedSource.out("update").to(finishedSink.in("update"));
    finishedSink.out("finished").asGraphOutput("request_finished");
    source.out("update").to(sink.in("update"));
    sink.out("finished").asGraphOutput("finished");
    if (!builder.ok()) return nullptr;
    navi::GraphBuilder::fillHashKey(graph.get());
    return graph;
}

void PPCacheCollectGraph::collect(const PPCacheShardEndpoint &endpoint,
                                  uint32_t shardCount)
{
    auto lastErrorLog = std::chrono::steady_clock::time_point{};
    while (true) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopping) return;
        }
        const auto generation = _merge->allocateGeneration();
        if (!_requestFinished->reserve(endpoint.shardId, generation)) return;
        auto graph = build(endpoint, shardCount, generation);
        auto closure = std::make_shared<ShardClosure>();
        navi::RunGraphParams params;
        params.setTimeoutMs(int64_t(10) * 365 * 24 * 60 * 60 * 1000);
        navi::ResourceMap resources;
        resources.addResource(_merge);
        resources.addResource(_requestFinished);
        auto result = graph ? _navi->runLocalGraphAsync(graph.release(), params,
                                                        resources, closure.get())
                            : nullptr;
        bool stopping;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _active[endpoint.shardId] = result;
            stopping = _stopping;
        }
        if (result) {
            if (stopping) result->abort(navi::EC_ABORT);
            closure->wait();
            const auto error = closure->error();
            const auto now = std::chrono::steady_clock::now();
            if (!error.empty() && now - lastErrorLog >= std::chrono::seconds(1)) {
                AUTIL_LOG(
                    WARN,
                    "PP shard stream ended, shard [%u], generation [%lu], error [%s]",
                    endpoint.shardId, generation, error.c_str());
                lastErrorLog = now;
            }
        }
        _requestFinished->disconnect(endpoint.shardId, generation);
        _merge->disconnect(endpoint.shardId, generation);
        std::unique_lock<std::mutex> lock(_mutex);
        _active[endpoint.shardId].reset();
        if (_condition.wait_for(lock, std::chrono::milliseconds(100),
                                [&]() { return _stopping; }))
            return;
    }
}

size_t PPCacheCollectGraph::publishRequestFinished(const PPShardRequestTerminal &finished)
{
    if (!_merge || !_requestFinished || !finished.request.requestId ||
        !finished.request.attemptId ||
        finished.status < PPShardRequestStatus::SUCCEEDED ||
        finished.status > PPShardRequestStatus::CANCELLED)
        return 0;
    size_t queued = 0;
    // Use the generations captured when this request began. Never queue its
    // notification onto a replacement connection or a disconnected shard.
    for (const auto &shard : _merge->pendingRequestShards(finished.request)) {
        if (_requestFinished->publish(shard.shardId, shard.generation, finished)) {
            ++queued;
        } else {
            _requestFinished->disconnect(shard.shardId, shard.generation);
            _merge->disconnect(shard.shardId, shard.generation);
        }
    }
    return queued;
}

void PPCacheCollectGraph::stop()
{
    std::vector<navi::NaviUserResultPtr> active;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopping = true;
        active = _active;
        _condition.notify_all();
    }
    for (const auto &result : active)
        if (result) result->abort(navi::EC_ABORT);
    for (auto &thread : _threads)
        if (thread.joinable()) thread.join();
    _threads.clear();
    _active.clear();
}

}
