#pragma once
#include "maga_transformer/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "maga_transformer/cpp/disaggregate/cache_store/TimerManager.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include <shared_mutex>

namespace rtp_llm {

class CacheStoreServiceImplContext:public std::enable_shared_from_this<CacheStoreServiceImplContext>{
public:
    CacheStoreServiceImplContext(const CacheLoadRequest* request, CacheLoadResponse* response, const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, ::google::protobuf::Closure* done);
    virtual ~CacheStoreServiceImplContext();

public:
    std::shared_ptr<BlockBufferInfo> getAndEraseUnLoadedBlock(const std::string& block_key);
    bool isAllLoaded();
    void setUnLoadedBlocks();
    void loadBlockOnTcp(std::shared_ptr<BlockBuffer> block);

    void setTimer(const std::shared_ptr<arpc::Timer> &timer) { timer_ = std::weak_ptr<arpc::Timer>(timer); }
    bool isTimeOut(){ return is_timeout_; }
    void setTimeOut(){ is_timeout_ = true; }
    void stopTimer();

    void runSuccess(bool direct_write);
    void runFailed(KvCacheStoreServiceErrorCode error_code);

    std::shared_ptr<std::atomic_bool> getAllSuccess() const {return all_success_;}
    std::shared_ptr<CacheStoreServerLoadMetricsCollector> getCollector()const {return collector_;}
    std::shared_ptr<std::atomic_int> getWriteCnt() const{return write_cnt_;}
    const CacheLoadRequest* getRequest() const{return request_;}

protected:
    const CacheLoadRequest* request_;
    
    std::mutex response_mutex_;
    CacheLoadResponse* response_;

    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;
    
    std::atomic_bool reentrant_flag_{false};
    ::google::protobuf::Closure* done_;

    std::weak_ptr<arpc::Timer> timer_;
    bool is_timeout_{false};
    
    std::shared_ptr<std::atomic_bool> all_success_;
    std::shared_ptr<std::atomic_int> write_cnt_;
    std::shared_mutex unloaded_blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBufferInfo>> unloaded_blocks_;
};
}