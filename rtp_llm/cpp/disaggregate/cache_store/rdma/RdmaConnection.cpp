#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"

#include "aios/network/accl-barex/include/accl/barex/xchannel.h"
#include "autil/EnvUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

RdmaConnection::RdmaConnection(const std::shared_ptr<MemoryUtil>&                              memory_util,
                               const std::string&                                              ip,
                               uint32_t                                                        port,
                               const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections):
    memory_util_(memory_util), ip_(ip), port_(port), connections_(connections) {
    enable_block_concat_ = autil::EnvUtil::getEnv(kEnvRdmaWriteBlockConcat, enable_block_concat_);
}

void RdmaConnection::writeBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& blocks, bool write_op) {
    if (enable_block_concat_) {
        // useful for a lot of blocks to write per layer
        auto concat_blocks = getConcatWriteBlocks(blocks);
        for (auto& block : concat_blocks) {
            doWriteBlock(block, write_op);
        }
    } else {
        doWriteBlocks(blocks, write_op);
    }
}

std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>
RdmaConnection::getConcatWriteBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& origin_blocks) {

    // make available blocks concat
    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>> sorted_blocks = origin_blocks;

    // sorted on mem pointer to concat
    std::sort(
        sorted_blocks.begin(),
        sorted_blocks.end(),
        [](const std::shared_ptr<RdmaConnectionWriteBlock>& lhs, const std::shared_ptr<RdmaConnectionWriteBlock>& rhs) {
            return lhs->getBlock()->addr.get() < rhs->getBlock()->addr.get();
        });

    // concat blocks
    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>> concat_blocks;
    std::shared_ptr<RdmaConnectionWriteBlock>              current_block;
    for (auto& block : sorted_blocks) {
        if (current_block != nullptr && current_block->appendBlock(block)) {
            continue;
        }
        current_block = block;
        concat_blocks.push_back(current_block);
    }

    return concat_blocks;
}

void RdmaConnection::doWriteBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& write_blocks, bool write_op) {
    auto start_time_us = currentTimeUs();

    std::map<std::shared_ptr<::accl::barex::XConnection>, std::vector<::accl::barex::rw_memp_t>> mems;
    std::map<std::shared_ptr<::accl::barex::XConnection>, std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>> blocks_map;

    bool isValid = true;
    for (auto& write_block: write_blocks) {
        auto& block = write_block->getBlock();
        auto& info  = write_block->getPeerBlock();

        ::accl::barex::rw_memp_t mem;
       
        if (!memory_util_->findMemoryMr((void*)&mem.data, block->addr.get(), block->len, block->gpu_mem, block->adopted)) {
            RTP_LLM_LOG_WARNING("find buffer mr for block %s failed", block->key.c_str());
            isValid = false;
            break;
        }

        auto connection = getConnection(write_block->getNicRkeys());
        if (connection == nullptr) {
            RTP_LLM_LOG_WARNING("get connection for %s:%u failed, not find matched nic rkey", ip_.c_str(), port_);
            isValid = false;
            break;
        }


        auto nic_rkeys = write_block->getNicRkeys();
        auto peer_nic  = connection->GetChannel()->GetPeerNicId();
        mem.r_addr     = info->rdma_info().addr();
        mem.r_key      = nic_rkeys[peer_nic];

        mems[connection].push_back(mem);
        blocks_map[connection].push_back(write_block);
    }

    if (!isValid) {
        for (auto& write_block : write_blocks) {
            write_block->getCallback()(false);
        }
        return;
    }

    for (auto& [connection, mem] : mems) {
        auto& write_blocks = blocks_map[connection];
        
        auto callback = [start_time_us, write_blocks, connection](::accl::barex::Status status) {
            if (!status.IsOk()) {
                auto peer_nic = connection->GetChannel()->GetPeerNicId();
                for (auto& write_block : write_blocks) {
                    auto& block = write_block->getBlock();
                    auto& info  = write_block->getPeerBlock();
                    auto  nic_rkeys = write_block->getNicRkeys();
                    auto  selected_rkey = nic_rkeys.count(peer_nic) ? nic_rkeys[peer_nic] : 0;
                    std::string rkey_str;
                    for (auto& kv : nic_rkeys) {
                        if (!rkey_str.empty()) rkey_str += ",";
                        rkey_str += std::to_string(kv.first) + ":" + std::to_string(kv.second);
                    }
                    RTP_LLM_LOG_WARNING("[RKEY_TRACE] prefill write failed: block=%s remote_addr=0x%lx "
                                        "remote_len=%u peer_nic=%u selected_rkey=%u nic_map=[%s] err=%s",
                                        block->key.c_str(),
                                        (unsigned long)info->rdma_info().addr(),
                                        block->len,
                                        peer_nic,
                                        selected_rkey,
                                        rkey_str.c_str(),
                                        status.ErrMsg().c_str());
                    write_block->getCallback()(false);
                }
                RTP_LLM_LOG_WARNING("write batch failed, err is %s", status.ErrMsg().c_str());
                return;
            }
            for (auto& write_block : write_blocks) {
                write_block->getCallback()(true);
            }
            RTP_LLM_LOG_DEBUG("write blocks done cost %ldus", currentTimeUs() - start_time_us);
        };

        if (write_op) {
            connection->GetChannel()->WriteBatch(mem, callback, false);
        } else {
            connection->GetChannel()->ReadBatch(mem, callback, false);
        }   
    }
}

void RdmaConnection::doWriteBlock(const std::shared_ptr<RdmaConnectionWriteBlock>& write_block, bool write_op) {
    auto start_time_us = currentTimeUs();

    auto& block = write_block->getBlock();
    auto& info  = write_block->getPeerBlock();
    auto& done  = write_block->getCallback();

    // verify local block valid
    ::accl::barex::rw_memp_t mem;
    if (!memory_util_->findMemoryMr((void*)&mem.data, block->addr.get(), block->len, block->gpu_mem, block->adopted)) {
        RTP_LLM_LOG_WARNING("find buffer mr for block %s failed", block->key.c_str());
        done(false);
        return;
    }

    // get valid connection
    auto connection = getConnection(write_block->getNicRkeys());
    if (connection == nullptr) {
        RTP_LLM_LOG_WARNING("get connection for %s:%u failed, not find matched nic rkey", ip_.c_str(), port_);
        done(false);
        return;
    }

    auto nic_rkeys = write_block->getNicRkeys();
    mem.r_addr     = info->rdma_info().addr();
    mem.r_key      = nic_rkeys[connection->GetChannel()->GetPeerNicId()];

    std::vector<::accl::barex::rw_memp_t> mems     = {mem};
    auto                                  callback = [start_time_us, write_block](::accl::barex::Status status) {
        if (!status.IsOk()) {
            RTP_LLM_LOG_WARNING("write batch failed, start block is %s, sub block count %u, err is %s",
                                write_block->getBlock()->key.c_str(),
                                write_block->getSubBlockCount(),
                                status.ErrMsg().c_str());
            write_block->getCallback()(false);
            return;
        }
        RTP_LLM_LOG_DEBUG("write block %s sub blocks %u done cost %ldus",
                          write_block->getBlock()->key.c_str(),
                          write_block->getSubBlockCount(),
                          currentTimeUs() - start_time_us);
        write_block->getCallback()(true);
    };

    if (write_op) {
        connection->GetChannel()->WriteBatch(mems, callback, false);
    } else {
        connection->GetChannel()->ReadBatch(mems, callback, false);
    }
}

std::shared_ptr<::accl::barex::XConnection>
RdmaConnection::getConnection(const std::map<uint32_t, uint32_t>& nic_rkeys) {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    if (connections_.empty()) {
        return nullptr;
    }
    for (int i = 0; i < connections_.size(); i++) {
        auto& connection  = connections_[write_count_++ % connections_.size()];
        auto  peer_nic_id = connection->GetChannel()->GetPeerNicId();
        if (nic_rkeys.find(peer_nic_id) != nic_rkeys.end()) {
            return connection;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<::accl::barex::XConnection>> RdmaConnection::getAllConnections() const {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    return connections_;
}

std::vector<std::pair<std::string, uint32_t>> RdmaConnection::getAllConnectionLocalAddrAndPort() const {
    std::vector<std::pair<std::string, uint32_t>> local_ip_port_vec;
    std::shared_lock<std::shared_mutex>           lock(connections_mutex_);
    for (auto connection : connections_) {
        auto local_ip_port = connection->GetLocalAddrAndPort();
        local_ip_port_vec.push_back(local_ip_port);
    }
    return local_ip_port_vec;
}

RdmaConnectionWriteBlock::RdmaConnectionWriteBlock(const std::shared_ptr<BlockBuffer>&     block,
                                                   const std::shared_ptr<BlockBufferInfo>& peer_block,
                                                   WriteBlockDoneCallback                  callback):
    block_(block), peer_block_(peer_block), callback_(callback) {
    for (int i = 0; i < peer_block->rdma_info().nic_rkeys_size(); i++) {
        nic_rkeys_[peer_block->rdma_info().nic_rkeys(i).nicid()] = peer_block->rdma_info().nic_rkeys(i).rkey();
    }
}

bool RdmaConnectionWriteBlock::appendBlock(const std::shared_ptr<RdmaConnectionWriteBlock>& append_block) {
    // mem should be same
    if (block_->gpu_mem != append_block->block_->gpu_mem || block_->adopted != append_block->block_->adopted) {
        return false;
    }

    // continuous mem
    if ((int64_t)(block_->addr.get()) + block_->len != (int64_t)(append_block->block_->addr.get())) {
        return false;
    }

    // peer continuous mem
    if (peer_block_->rdma_info().addr() + peer_block_->len() != append_block->peer_block_->rdma_info().addr()) {
        return false;
    }

    // nic_rkeys should be same
    if (nic_rkeys_ != append_block->nic_rkeys_) {
        return false;
    }

    if (sub_blocks_.empty()) {
        // first time, save this block to sub
        sub_blocks_.push_back(
            std::make_shared<RdmaConnectionWriteBlock>(block_, peer_block_, callback_));  // save origin block

        // update current block with copy
        block_      = std::make_shared<BlockBuffer>(*block_);
        peer_block_ = std::make_shared<BlockBufferInfo>(*peer_block_);
    }

    block_->len += append_block->block_->len;
    peer_block_->set_len(peer_block_->len() + append_block->peer_block_->len());
    sub_blocks_.push_back(append_block);

    callback_ = [sub_blocks = sub_blocks_](bool ok) {
        // copy once, avoid cycle ref
        for (auto& sub_block : sub_blocks) {
            sub_block->callback_(ok);
        }
    };
    return true;
}

size_t RdmaConnection::getConnectionsCount() const {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    return connections_.size();
}

void RdmaConnection::addConnection(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    std::unique_lock<std::shared_mutex> lock(connections_mutex_);
    connections_.push_back(connection);
}

}  // namespace rtp_llm