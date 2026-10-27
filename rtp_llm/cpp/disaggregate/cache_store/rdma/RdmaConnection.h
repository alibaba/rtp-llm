#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {

class RdmaConnectionWriteBlock {
public:
    RdmaConnectionWriteBlock(const std::shared_ptr<BlockBuffer>&     block_,
                             const std::shared_ptr<BlockBufferInfo>& peer_block_,
                             WriteBlockDoneCallback                  callback_);

    bool appendBlock(const std::shared_ptr<RdmaConnectionWriteBlock>& block);

    const std::shared_ptr<BlockBuffer>& getBlock() const {
        return block_;
    }
    const std::shared_ptr<BlockBufferInfo>& getPeerBlock() const {
        return peer_block_;
    }
    const WriteBlockDoneCallback& getCallback() const {
        return callback_;
    }
    const std::map<uint32_t, uint32_t>& getNicRkeys() const {
        return nic_rkeys_;
    }
    uint32_t getSubBlockCount() const {
        return sub_blocks_.size();
    }

private:
    std::shared_ptr<BlockBuffer>                           block_;
    std::shared_ptr<BlockBufferInfo>                       peer_block_;
    WriteBlockDoneCallback                                 callback_;
    std::map<uint32_t, uint32_t>                           nic_rkeys_;
    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>> sub_blocks_;
};

// RdmaConnection used to wrap connection on rdma
// Each RdmaConnection should refs to same peer process
// RdmaConnection may contains multiple real connections
// Each connection may locate in diffrent nic
class RdmaConnection {
public:
    RdmaConnection(const std::shared_ptr<MemoryUtil>&                              memory_util,
                   const std::string&                                              ip,
                   uint32_t                                                        port,
                   const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections);

public:
    void   addConnection(const std::shared_ptr<::accl::barex::XConnection>& connection);
    size_t getConnectionsCount() const;
    std::vector<std::shared_ptr<::accl::barex::XConnection>> getAllConnections() const;
    std::vector<std::pair<std::string, uint32_t>>            getAllConnectionLocalAddrAndPort() const;

    void writeBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& blocks, bool write_op = true);

    const std::string& getIp() const {
        return ip_;
    }

    uint32_t getPort() const {
        return port_;
    }

private:
    void doWriteBlock(const std::shared_ptr<RdmaConnectionWriteBlock>& block, bool write_op);
    void doWriteBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& blocks, bool write_op);

    std::shared_ptr<::accl::barex::XConnection> getConnection(const std::map<uint32_t, uint32_t>& nic_rkeys);
    std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>
    getConcatWriteBlocks(const std::vector<std::shared_ptr<RdmaConnectionWriteBlock>>& blocks);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    std::string                 ip_;
    uint32_t                    port_;
    bool                        enable_block_concat_{false};

    std::atomic_int                                          write_count_{0};
    mutable std::shared_mutex                                connections_mutex_;
    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections_;
};

}  // namespace rtp_llm