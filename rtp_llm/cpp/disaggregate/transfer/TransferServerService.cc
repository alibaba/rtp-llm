#include "rtp_llm/cpp/disaggregate/transfer/TransferServerService.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TransferServerService::TransferServerService(
    const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store,
    const std::shared_ptr<LayerBlockConvertor>&       layer_block_convector,
    rtp_llm::DeviceBase*                              device,
    const std::shared_ptr<IRdmaClient>&               rdma_client):
    layer_cache_buffer_task_store_(layer_cache_buffer_task_store),
    layer_block_convector_(layer_block_convector),
    device_(device),
    rdma_client_(rdma_client) {}

TransferServerService::~TransferServerService() {}

void TransferServerService::transfer(::google::protobuf::RpcController*           controller,
                                     const ::transfer::LayerBlockTransferRequest* request,
                                     ::transfer::LayerBlockTransferResponse*      response,
                                     ::google::protobuf::Closure*                 done) {
    RTP_LLM_LOG_INFO("TransferServerService transfer start, unique_key: %s, layer_id: %d",
                     request->unique_key().c_str(),
                     request->layer_block().layer_id());
    auto run_failed_func = [done, response](const std::string& info) {
        RTP_LLM_LOG_WARNING("transfer failed, info: %s", info.c_str());
        response->set_success(false);
        response->set_info(info);
        done->Run();
    };

    // 检查是否有 layer_block
    if (!request->has_layer_block() || !request->has_partition_count() || !request->has_partition_id()
        || !request->has_unique_key()) {
        run_failed_func("request has no layer_block or partition_count or partition_id or context_id");
        return;
    }

    auto unique_key       = request->unique_key();
    auto layer_block_info = request->layer_block();
    int  layer_id         = layer_block_info.layer_id();

    // 查找对应的 LayerCacheBuffer
    auto                                  start_time_ms = currentTimeMs();
    std::shared_ptr<LayerCacheBufferTask> task;
    auto                                  deadline_ms = currentTimeMs() + 1000;
    // TODO: async wait
    while (currentTimeMs() < deadline_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        task = layer_cache_buffer_task_store_->getTask(unique_key);
        if (task) {
            break;
        }
    }
    RTP_LLM_LOG_INFO("TransferServerService transfer wait task time: %lld ms", currentTimeMs() - start_time_ms);

    if (!task) {
        run_failed_func("layer cache buffer task not found, unique_key: " + unique_key);
        return;
    }

    auto layer_cache_buffer = task->getLayerCacheBuffer(layer_id);
    if (!layer_cache_buffer) {
        run_failed_func("layer cache buffer not found, unique_key: " + unique_key
                        + ", layer_id: " + std::to_string(layer_id));
        return;
    }

    // TODO: test this
    if (task->cancelled() || !task->success()) {
        // failed task no need to transfer, treat as transfer failed
        task->notifyDone(layer_id, false);
        run_failed_func("layer cache buffer task cancelled or failed, unique_key: " + unique_key);
        return;
    }

    task->setLoading(layer_id);
    if (rdma_client_) {
        transferViaRdma(controller, request, response, done, layer_cache_buffer, task);
        return;
    }

    // 使用 TCP 模式
    transferViaTcp(controller, request, response, done, layer_cache_buffer, task);
}

void TransferServerService::transferViaTcp(::google::protobuf::RpcController*           controller,
                                           const transfer::LayerBlockTransferRequest*   request,
                                           ::transfer::LayerBlockTransferResponse*      response,
                                           ::google::protobuf::Closure*                 done,
                                           const std::shared_ptr<LayerCacheBuffer>&     layer_cache_buffer,
                                           const std::shared_ptr<LayerCacheBufferTask>& task) {
    RTP_LLM_LOG_INFO("TransferServerService transferViaTcp start, unique_key: %s, layer_id: %d",
                     request->unique_key().c_str(),
                     request->layer_block().layer_id());
    auto run_failed_func =
        [done, response, task, layer_id = layer_cache_buffer->getLayerId()](const std::string& info) {
            task->notifyDone(layer_id, false);
            RTP_LLM_LOG_WARNING("transfer via tcp failed, info: %s", info.c_str());
            response->set_success(false);
            response->set_info(info);
            done->Run();
        };

    auto layer_block_info = request->layer_block();
    int  partition_count  = request->partition_count();
    int  partition_id     = request->partition_id();
    auto layer_id         = layer_cache_buffer->getLayerId();

    // 从 request 中获取 block 信息，并写入到 layer_cache_buffer 对应的 buffer 中
    int transfer_count = 0;
    for (const auto& block_info : layer_block_info.blocks()) {
        int64_t key      = block_info.key();
        auto    block_id = layer_cache_buffer->getBlockId(key);
        if (block_id == -1) {
            // no need to transfer
            continue;
        }

        // 使用 LayerBlockConvertor 获取对应的 buffer
        auto buffers = layer_block_convector_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        if (buffers.size() != block_info.blocks_size()) {
            run_failed_func("buffers mismatch, layer_id: " + std::to_string(layer_id)
                            + ", block_id: " + std::to_string(block_id));
            return;
        }

        for (int i = 0; i < buffers.size(); i++) {
            auto buffer            = buffers[i];
            auto block_buffer_info = block_info.blocks(i);
            auto src_buffer        = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU,
                                              rtp_llm::DataType::TYPE_UINT8,
                                                     {block_buffer_info.len()},
                                              block_buffer_info.content().data());
            device_->noBlockCopy({*buffer, src_buffer});
        }
        transfer_count++;
    }

    // 检查是否所有 block 都已成功传输
    if (transfer_count != layer_cache_buffer->blockIdMap().size()) {
        run_failed_func("transfer count mismatch, layer_id: " + std::to_string(layer_id)
                        + ", expected: " + std::to_string(layer_cache_buffer->blockIdMap().size())
                        + ", actual: " + std::to_string(transfer_count));
        return;
    }

    task->notifyDone(layer_id, true);
    response->set_success(true);
    done->Run();
}

void TransferServerService::transferViaRdma(::google::protobuf::RpcController*           controller,
                                            const ::transfer::LayerBlockTransferRequest* request,
                                            ::transfer::LayerBlockTransferResponse*      response,
                                            ::google::protobuf::Closure*                 done,
                                            const std::shared_ptr<LayerCacheBuffer>&     layer_cache_buffer,
                                            const std::shared_ptr<LayerCacheBufferTask>& task) {
    RTP_LLM_LOG_INFO("TransferServerService transferViaRdma start, unique_key: %s, layer_id: %d",
                     request->unique_key().c_str(),
                     request->layer_block().layer_id());
    auto run_failed_func =
        [done, response, task, layer_id = layer_cache_buffer->getLayerId()](const std::string& info) {
            task->notifyDone(layer_id, false);
            RTP_LLM_LOG_WARNING("transfer via rdma failed, info: %s", info.c_str());
            response->set_success(false);
            response->set_info(info);
            done->Run();
        };

    // 从 request 中获取 server 的 RDMA IP 和 port
    std::string server_ip   = request->server_rdma_ip();
    uint32_t    server_port = request->server_rdma_port();

    if (server_ip.empty() || server_port == 0) {
        run_failed_func("request has no rdma ip or port");
        return;
    }

    auto layer_block_info = request->layer_block();
    int  partition_count  = request->partition_count();
    int  partition_id     = request->partition_id();
    auto layer_id         = layer_cache_buffer->getLayerId();

    if (layer_cache_buffer->blockIdMap().size() != layer_block_info.blocks_size()) {
        run_failed_func("layer cache buffer block id map size mismatch, layer_id: " + std::to_string(layer_id)
                        + ", expected: " + std::to_string(layer_cache_buffer->blockIdMap().size())
                        + ", actual: " + std::to_string(layer_block_info.blocks_size()));
        return;
    }

    auto convert_to_remote_buffer_func =
        [](const transfer::BlockBufferInfo& block_buffer_info) -> std::shared_ptr<RemoteBuffer> {
        const auto& rdma_info = block_buffer_info.rdma_info();
        auto        nic_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();

        for (const auto& nic_rkey : rdma_info.nic_rkeys()) {
            nic_rkeys->insert({nic_rkey.nicid(), nic_rkey.rkey()});
        }
        return std::make_shared<RemoteBuffer>(
            static_cast<int64_t>(rdma_info.addr()), block_buffer_info.len(), nic_rkeys);
    };

    // 收集所有需要 RDMA read 的 buffer 对
    int                                                              transfer_count = 0;
    std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>> local_remote_buffers;
    for (const auto& block_info : layer_block_info.blocks()) {
        int64_t key      = block_info.key();
        auto    block_id = layer_cache_buffer->getBlockId(key);
        if (block_id == -1) {
            continue;
        }

        // 使用 LayerBlockConvertor 获取对应的 buffer
        auto buffers = layer_block_convector_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        if (buffers.size() != block_info.blocks_size()) {
            run_failed_func("buffers mismatch");
            return;
        }

        for (int i = 0; i < buffers.size(); i++) {
            auto remote_buffer = convert_to_remote_buffer_func(block_info.blocks(i));
            local_remote_buffers.push_back({buffers[i], remote_buffer});
        }
        transfer_count++;
    }

    if (local_remote_buffers.empty() || transfer_count != static_cast<int>(layer_cache_buffer->blockIdMap().size())) {
        RTP_LLM_LOG_WARNING(
            "TransferServerService transferViaRdma no buffers to read via RDMA transfer count: %d, layer_cache_buffer block id map size: %d",
            transfer_count,
            layer_cache_buffer->blockIdMap().size());
        run_failed_func("no buffers to read via RDMA");
        return;
    }

    RTP_LLM_LOG_INFO("TransferServerService transferViaRdma local_remote_buffers size: %ld, transfer_count: %d",
                     local_remote_buffers.size(),
                     transfer_count);
    // 获取 RDMA 连接
    auto connection = rdma_client_->getConnection(server_ip, server_port);
    if (!connection) {
        run_failed_func("get RDMA connection failed, ip: " + server_ip + ", port: " + std::to_string(server_port));
        return;
    }

    // 调用 RDMA read
    connection->read(
        local_remote_buffers,
        [done,
         response,
         run_failed_func,
         task,
         layer_id   = layer_cache_buffer->getLayerId(),
         unique_key = request->unique_key()](bool success) {
            RTP_LLM_LOG_INFO(
                "TransferServerService transferViaRdma read done, unique_key: %s, layer_id: %d, success: %d",
                unique_key.c_str(),
                layer_id,
                success);
            if (!success) {
                run_failed_func("RDMA read failed");
                return;
            }

            task->notifyDone(layer_id, true);
            response->set_success(true);
            done->Run();
        });
}

}  // namespace rtp_llm
