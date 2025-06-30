#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"

#include "autil/NetUtil.h"

namespace rtp_llm {

CacheLoadRequest* Messager::makeLoadRequest(const std::shared_ptr<LoadRequest>& request) {
    auto blocks = request->request_block_buffer->getBlocks();

    auto load_request = new CacheLoadRequest;
    load_request->set_timeout_ms(request->timeout_ms - 10);
    load_request->set_requestid(request->request_block_buffer->getRequestId());
    load_request->set_client_ip(autil::NetUtil::getBindIp());
    load_request->set_request_send_start_time_us(autil::TimeUtility::currentTimeInMicroSeconds());
    load_request->set_partition_count(request->partition_count);
    load_request->set_partition_id(request->partition_id);

    for (auto& [key, block] : blocks) {
        auto block_msg = load_request->add_blocks();
        block_msg->set_key(block->key);
        block_msg->set_len(block->len);
    }
    return load_request;
}

}  // namespace rtp_llm