#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreLoadServiceClosure.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/IndexerCacheScale.h"

namespace rtp_llm {

RdmaCacheStoreLoadServiceClosure::RdmaCacheStoreLoadServiceClosure(
    const std::shared_ptr<RdmaClient>&                           rdma_client,
    const std::shared_ptr<RdmaConnection>&                       rdma_connection,
    const std::shared_ptr<MemoryUtil>&                           memory_util,
    const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
    arpc::ANetRPCController*                                     controller,
    CacheLoadRequest*                                            request,
    CacheLoadResponse*                                           response,
    CacheStoreLoadDoneCallback                                   callback,
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector):
    rdma_client_(rdma_client),
    rdma_connection_(rdma_connection),
    memory_util_(memory_util),
    request_block_buffer_(request_block_buffer),
    controller_(controller),
    request_(request),
    response_(response),
    callback_(callback),
    collector_(collector) {}

RdmaCacheStoreLoadServiceClosure::~RdmaCacheStoreLoadServiceClosure() {
    recycleRdmaConnection(false);
    if (controller_) {
        delete controller_;
    }
    if (request_) {
        delete request_;
    }
    if (response_) {
        delete response_;
    }
}

void RdmaCacheStoreLoadServiceClosure::Run() {
    collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("cache load request failed, controller err is %d", controller_->GetErrorCode());
        recycleRdmaConnection(false);
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("cache load request failed, response err is %d", response_->error_code());
        recycleRdmaConnection(false);
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    try {
        std::unordered_map<std::string, const BlockBufferInfo*> descriptors;
        for (const auto& info : response_->blocks()) {
            if (!descriptors.emplace(info.key(), &info).second) {
                throw std::runtime_error("duplicate indexer block metadata");
            }
        }
        std::vector<IndexerScaleConversion> conversions;
        for (const auto& [key, block] : request_block_buffer_->getBlocks()) {
            const auto it = descriptors.find(key);
            if (it == descriptors.end()) {
                conversions.push_back({block->blockInfo(), {}});
                continue;
            }
            if (it->second->len() != block->len) {
                throw std::runtime_error("missing or invalid indexer block metadata");
            }
            conversions.push_back({block->blockInfo(), IndexerCacheLayout::fromWire(*it->second)});
        }
        convertIndexerCacheScales(conversions);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("indexer cache conversion failed: %s", e.what());
        recycleRdmaConnection(false);
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
        return;
    }
    recycleRdmaConnection(true);
    end(true, CacheStoreErrorCode::None);
}

// force close rdma connection to make safe write
void RdmaCacheStoreLoadServiceClosure::recycleRdmaConnection(bool success) {
    if (rdma_connection_ == nullptr) {
        return;
    }
    auto rdma_client = rdma_client_.lock();
    if (rdma_client) {
        if (success) {
            rdma_client->recycleConnection(rdma_connection_);
        } else {
            rdma_client->destroyConnection(rdma_connection_);
        }
        rdma_connection_.reset();
    }
}

void RdmaCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm