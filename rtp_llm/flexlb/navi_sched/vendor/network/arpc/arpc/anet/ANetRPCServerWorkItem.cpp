#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCServerWorkItem.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCServerClosure.h"

using namespace std;

ARPC_BEGIN_NAMESPACE(arpc)

ARPC_DECLARE_AND_SETUP_LOGGER(ANetRPCServerWorkItem);

ANetRPCServerWorkItem::ANetRPCServerWorkItem(RPCService *pService,
                                             RPCMethodDescriptor *pMethodDes,
                                             RPCMessage *pReqMsg,
                                             const std::string &userPayload,
                                             anet::Connection *pConnection,
                                             channelid_t channelId,
                                             ANetRPCMessageCodec *messageCodec,
                                             version_t version,
                                             Tracer *tracer)
    : RPCServerWorkItem(pService, pMethodDes, pReqMsg, userPayload, tracer)
    , _channelId(channelId)
    , _version(version)
    , _messageCodec(messageCodec) {
    _pConnection = pConnection;
    _pConnection->addRef();
}

ANetRPCServerWorkItem::~ANetRPCServerWorkItem() { _pConnection->subRef(); }

void ANetRPCServerWorkItem::destroy() {
    if (_tracer) {
        DELETE_AND_SET_NULL_ANET(_tracer);
    }
    delete this;
}

string ANetRPCServerWorkItem::getClientAddress() {
    string clientAddress;
    anet::IOComponent *ioComponent = NULL;

    if (NULL != (ioComponent = _pConnection->getIOComponent())) {
        anet::Socket *socket = ioComponent->getSocket();

        if (socket) {
            char dest[64];
            memset(dest, 0, sizeof(dest[0]) * 30);

            if (socket->getAddr(dest, 64, false) != NULL) {
                clientAddress.assign(dest);
                return clientAddress;
            }
        }
    }

    clientAddress.assign("unknownAddress");
    return clientAddress;
}

void ANetRPCServerWorkItem::process() {
    _tracer->BeginWorkItemProcess();
    assert(_arena);
    RPCMessage *pResMsg = _pService->GetResponsePrototype(_pMethodDes).New(_arena.get());
    RPCServerClosure *pClosure =
        new (nothrow) ANetRPCServerClosure(_channelId, _pConnection, _pReqMsg, pResMsg, _messageCodec, _version);

    if (pClosure == NULL) {
        ARPC_LOG(ERROR, "new RPCServerClosure return NULL");
        return;
    }

    _tracer->setUserPayload(_userPayload);
    pClosure->setProtobufArena(_arena);
    pClosure->SetTracer(_tracer);
    _tracer = nullptr;
    RPCController *pController = pClosure->GetRpcController();

    ANetRPCController *anetRpcController = dynamic_cast<ANetRPCController *>(pController);

    if (anetRpcController) {
        string clientAddress = getClientAddress();
        anetRpcController->SetClientAddress(clientAddress);
    }

    _pService->CallMethod(_pMethodDes, pController, _pReqMsg, pResMsg, pClosure);

    return;
}

ARPC_END_NAMESPACE(arpc)
