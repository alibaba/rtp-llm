#include "aios/navi_sched/data/FlexlbScheduleData.h"
#include "navi/engine/Kernel.h"
#include <grpc++/generic/generic_stub.h>
#include <grpc++/client_context.h>

namespace navi_sched {
namespace {

struct FlexlbRpcCall {
    std::shared_ptr<grpc::ClientContext> context;
    std::unique_ptr<grpc::GenericStub> stub;
    grpc::ByteBuffer request;
    grpc::ByteBuffer response;
};

}

class ClientFlexlbKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.client_flexlb_k")
            .output("response", FLEXLB_RESPONSE_TYPE);
    }

    navi::ErrorCode init(navi::KernelInitContext &ctx) override
    {
        if (!_request || !_request->context || !_request->channel) {
            return navi::EC_INIT_SUB_GRAPH;
        }
        _pipe = ctx.createAsyncPipe();
        if (!_pipe) {
            return navi::EC_INIT_SUB_GRAPH;
        }
        _call = std::make_shared<FlexlbRpcCall>();
        _call->context = _request->context;
        _call->stub = std::make_unique<grpc::GenericStub>(_request->channel);
        grpc::Slice slice(_request->payload);
        _call->request = grpc::ByteBuffer(&slice, 1);
        auto call = _call;
        auto pipe = _pipe;
        call->stub->experimental().UnaryCall(
            call->context.get(), "/FlexlbService/Schedule",
            &call->request, &call->response,
            [call, pipe](grpc::Status status) {
                auto response = std::make_shared<FlexlbScheduleResponseData>();
                response->statusCode = static_cast<int32_t>(status.error_code());
                response->errorMessage = status.error_message();
                if (status.ok()) {
                    std::vector<grpc::Slice> slices;
                    auto dumped = call->response.Dump(&slices);
                    if (!dumped.ok()) {
                        response->statusCode = static_cast<int32_t>(grpc::StatusCode::INTERNAL);
                        response->errorMessage = "decode FlexLB response buffer failed";
                    } else {
                        for (const auto &part : slices) {
                            response->payload.append(
                                reinterpret_cast<const char *>(part.begin()), part.size());
                        }
                    }
                }
                pipe->setData(response);
                pipe->setEof();
            });
        return navi::EC_NONE;
    }

    ~ClientFlexlbKernel() override
    {
        if (_call) {
            _call->context->TryCancel();
        }
    }

    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr response;
        bool eof = false;
        if (_pipe->getData(response, eof) && response) {
            ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), response, true);
        }
        return navi::EC_NONE;
    }

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_NAMED_DATA(FlexlbScheduleRequestData, _request, FLEXLB_REQUEST_NAME);
    navi::AsyncPipePtr _pipe;
    std::shared_ptr<FlexlbRpcCall> _call;
};

REGISTER_KERNEL(ClientFlexlbKernel);

}
