#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTask.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"

namespace rtp_llm {

class CacheTransferServiceClosure: public RPCClosure {
public:
    CacheTransferServiceClosure(const std::shared_ptr<TransferRequest>& transfer_request,
                                CacheTransferRequest*                   request,
                                CacheTransferResponse*                  response,
                                arpc::ANetRPCController*                controller):
        transfer_request_(transfer_request), request_(request), response_(response), controller_(controller) {}
    ~CacheTransferServiceClosure();

public:
    void Run() override;

private:
    void end(bool success, CacheStoreErrorCode error_code);

protected:
    std::shared_ptr<TransferRequest> transfer_request_;
    CacheTransferRequest*            request_{nullptr};
    CacheTransferResponse*           response_{nullptr};
    arpc::ANetRPCController*         controller_{nullptr};
};

}  // namespace rtp_llm