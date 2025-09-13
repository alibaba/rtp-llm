#pragma once

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <list>
#include <tuple>
#include <functional>
#include <sstream>

namespace rtp_llm {
class DeviceBase;

class BatchState {
public:
    BatchState() = default;

    BatchState(size_t numCtxRequests, size_t numGenRequests, size_t numTokens, size_t maxKvCacheLength):
        mNumCtxRequests{numCtxRequests},
        mNumGenRequests{numGenRequests},
        mNumTokens{numTokens},
        mMaxKvCacheLength{maxKvCacheLength} {}

    bool hasPrefill() const {
        return mNumCtxRequests > 0;
    }

    bool operator==(BatchState const& other) const {
        return mNumCtxRequests == other.mNumCtxRequests && mNumGenRequests == other.mNumGenRequests
               && mNumTokens == other.mNumTokens && mMaxKvCacheLength == other.mMaxKvCacheLength;
    }

    size_t hash() const {
        size_t h1 = std::hash<size_t>{}(mNumCtxRequests);
        size_t h2 = std::hash<size_t>{}(mNumGenRequests);
        size_t h3 = std::hash<size_t>{}(mNumTokens);
        size_t h4 = std::hash<size_t>{}(mMaxKvCacheLength);
        return h1 ^ h2 ^ h3 ^ h4;
    }
    std::string debugString() const {
        std::stringstream ss;
        ss << "BatchState ( "
           << "mNumCtxRequests=" << mNumCtxRequests << ", "
           << "mNumGenRequests=" << mNumGenRequests << ", "
           << "mNumTokens=" << mNumTokens << ", "
           << "mMaxKvCacheLength=" << mMaxKvCacheLength << ", "
           << " )";
        return ss.str();
    }

    size_t mNumCtxRequests;
    size_t mNumGenRequests;
    size_t mNumTokens;         // TODO: reserved
    size_t mMaxKvCacheLength;  // TODO: reserved
};

struct BatchStateHash {
    size_t operator()(BatchState const& bs) const {
        return bs.hash();
    }
};

class ExecutorBase {
public:
    virtual void replay()       = 0;
    virtual void captureBegin() = 0;
    virtual void captureEnd()   = 0;
};

template<typename Input, typename Output>
class CudaGraphExecutorCache {
    /// @brief LRU cache to store cuda graph instances.
public:
    using GraphItem             = std::tuple<BatchState, std::shared_ptr<ExecutorBase>, Input, Output>;
    using GraphExecutorLruCache = std::list<GraphItem>;

    explicit CudaGraphExecutorCache(size_t capacity): mCapacity(capacity) {}

    std::optional<GraphItem> get(const BatchState& state);
    void                     put(const BatchState& state, const GraphItem& item);
    std::string              debugString(const std::string& hint = "") const {
        std::stringstream s;
        RTP_LLM_CHECK_WITH_INFO(
            mCache.size() == mMap.size(), "mCache.size() = %d but mMap.size() = %d", mCache.size(), mMap.size());
        if (!hint.empty())
            s << "hint: " << hint << std::endl;
        s << "size: " << mCache.size() << std::endl;
        for (auto& item : mCache) {
            s << std::string(100, '@') << std::endl;
            s << "batch : " << std::get<0>(item).debugString() << std::endl;
            s << "executor : " << &std::get<1>(item) << std::endl;
            s << "input : " << &std::get<2>(item) << std::endl;
            s << "output : " << &std::get<3>(item) << std::endl;
            s << std::string(100, '#') << std::endl;
        }
        return s.str();
    }

private:
    size_t                                                                                   mCapacity;
    GraphExecutorLruCache                                                                    mCache;
    std::unordered_map<BatchState, typename GraphExecutorLruCache::iterator, BatchStateHash> mMap;
};

template<typename Input, typename Output>
class NativeGraphRunnerBase {
public:
    NativeGraphRunnerBase(DeviceBase* device);
    Output run(size_t prefill_bs, size_t decode_bs, Input input, std::function<Output(Input)> forward);
    Input  prepareInputBuffer(const Input& old) {
        static_assert(sizeof(Input) == 0, "Type of Input not supported for prepareInputBuffer");
        return old;
    }
    void copy(Input* dst, const Input& src) {
        static_assert(sizeof(Input) == 0, "Type of Input not supported for copy");
    }

protected:
    virtual std::shared_ptr<ExecutorBase> makeExecutor() = 0;

protected:
    DeviceBase* device_;

private:
    std::unique_ptr<CudaGraphExecutorCache<Input, Output>> map_;
};

#define INSTANTIATE_NATIVE_GRAPH_RUNNER_BASE(I, O) template class NativeGraphRunnerBase<I, O>;

}  // namespace rtp_llm