#include "aios/network/rdma/accl/unittest/FakeRdmaServerAdapter.h"

#include "aios/network/rdma/accl/RdmaChannelServer.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc)

int FakeRdmaServerAdapter::HandlePacket(const std::shared_ptr<RdmaMessage>&       message,
                                        const std::shared_ptr<RdmaChannelServer>& channel,
                                        const std::shared_ptr<ServerRPCStats>&    stats) {
    auto request = std::make_shared<FakeRdmaRequest>();
    EXPECT_TRUE(request->Decode(message));

    request->SetValue(request->GetValue() + "-response");

    auto responseMessage = request->Encode();
    channel->SendNormalResponse(request->GetConnection(), responseMessage, stats);
    return 0;
}

ARPC_END_NAMESPACE(arpc)