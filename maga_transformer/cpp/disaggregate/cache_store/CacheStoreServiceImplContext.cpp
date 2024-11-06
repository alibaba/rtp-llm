#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <atomic>

#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

CacheStoreServiceImplContext::CacheStoreServiceImplContext(const CacheLoadRequest* request, CacheLoadResponse* response, const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, ::google::protobuf::Closure* done)
:request_(request), response_(response), collector_(collector), done_(done), all_success_(std::make_shared<std::atomic_bool>(true)), write_cnt_(std::make_shared<std::atomic_int>(0)){}

CacheStoreServiceImplContext::~CacheStoreServiceImplContext(){}
void CacheStoreServiceImplContext::stopTimer(){
    if (auto timer_shared_ptr = timer_.lock()) {
        timer_shared_ptr->stop();
        timer_shared_ptr.reset();
    }
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key){
    std::shared_ptr<BlockBufferInfo> block_info;
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto it = unloaded_blocks_.find(block_key);
    if(it == unloaded_blocks_.end()){
        return nullptr;
    }

    block_info = it->second;
    unloaded_blocks_.erase(it);
    return block_info; 
}
void CacheStoreServiceImplContext::runSuccess(bool direct_write){
    FT_LOG_DEBUG("run success");
    bool expected = false;
    if(!reentrant_flag_.compare_exchange_strong(expected,true)){
        return ;
    }
    stopTimer();
    CacheStoreServerLoadMetricsCollector::markEnd(collector_, true);
    response_mutex_.lock();
    response_->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
    response_->set_response_send_start_time_us(autil::TimeUtility::currentTimeInMicroSeconds());
    response_->set_direct_write_response(direct_write);
    if(done_){ 
        done_->Run();
        done_=nullptr;
    }
    response_=nullptr;
    response_mutex_.unlock();
}

void CacheStoreServiceImplContext::runFailed(KvCacheStoreServiceErrorCode error_code){
    bool expected = false;
    if(!reentrant_flag_.compare_exchange_strong(expected,true)){
        return ;
    }
    stopTimer();
    FT_LOG_WARNING(
                  "cache store service load failed, request %s from [%s], error code is %d",
                  request_->requestid().c_str(),
                  request_->client_ip().c_str(),
                  error_code);
    CacheStoreServerLoadMetricsCollector::markEnd(collector_, false);
    response_mutex_.lock();
    response_->clear_blocks();
    response_->set_error_code(error_code);
    if(done_){     
        done_->Run();
        done_=nullptr;
    }
    response_=nullptr;
    response_mutex_.unlock();
}

void CacheStoreServiceImplContext::setUnLoadedBlocks(){
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    for (int i = 0; i < request_->blocks_size(); i++) {
        unloaded_blocks_[request_->blocks(i).key()] = std::make_shared<BlockBufferInfo>(request_->blocks(i));
    }
}

bool CacheStoreServiceImplContext::isAllLoaded(){
    return write_cnt_->load()==request_->blocks_size();
}

void CacheStoreServiceImplContext::loadBlockOnTcp(std::shared_ptr<BlockBuffer> block){
    auto unloaded_block_info = getAndEraseUnLoadedBlock(block->key);
    if(unloaded_block_info == nullptr){
        return ;
    }
    
    if(unloaded_block_info->len()!=block->len){
        runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
        return ; 
    }

    response_mutex_.lock();
    if(response_ == nullptr){
        response_mutex_.unlock();
        return ;
    }
    auto* block_info = response_->add_blocks();
    block_info->set_key(block->key);
    block_info->set_len(block->len);
    auto block_content = block_info->mutable_content();
    block_content->assign(
        std::shared_ptr<const char>(block->addr, reinterpret_cast<const char*>(block->addr.get())),
        size_t(block->len));
    response_mutex_.unlock();

    if (++(*write_cnt_) == 1) {
        CacheStoreServerLoadMetricsCollector::setFirstBlockCostUs(
            collector_, autil::TimeUtility::currentTimeInMicroSeconds() - request_->request_send_start_time_us());
    }
    
    FT_LOG_DEBUG("load in callback %s", block->key.c_str());
    if(all_success_->load() && isAllLoaded()){
        runSuccess(false);
    } 
}

}